"""
@ Description: Protein to DGL graph with node and edge features
"""

import os, copy
import dgl
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Union
from joblib import Parallel, delayed
import pandas as pd
from sklearn.model_selection import train_test_split
from graph_transformer_v2 import Gate
import lightning as L
from torch.utils.data import Dataset
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger
import wandb
from scipy.stats.stats import pearsonr
import scipy.sparse as sp
 
class DGLData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str, label_folder: str, targets):
        self.target_list = targets
        self.dgl_folder = dgl_folder
        self.label_folder = label_folder

        self.data = []
        self.node_label = []
        # self.edge_label = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.node_label[idx]#, self.edge_label[idx]

    def _prepare(self):
        for target in self.target_list:
            target_dgl_folder = self.dgl_folder + '/' + target
            target_label_folder = self.label_folder + '/' + target

            for dgl_file in os.listdir(target_dgl_folder):
                g, tmp = dgl.data.utils.load_graphs(target_dgl_folder + '/' + dgl_file)
                self.data.append(g[0])
                self.node_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_node.npy')))
                # self.edge_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_edge.npy')))


def collate(samples):
    """Customer collate function"""
    graphs, node_labels = zip(*samples)
    batched_graphs = dgl.batch(graphs)
    batch_node_labels, batch_edge_labels = None, None
    for node_label in node_labels:
        if batch_node_labels is None:
            batch_node_labels = copy.deepcopy(node_label)
        else:
            batch_node_labels = np.concatenate((batch_node_labels, node_label), axis=None)
    
    # for edge_label in edge_labels:
    #     if batch_edge_labels is None:
    #         batch_edge_labels = copy.deepcopy(edge_label)
    #     else:
    #         batch_edge_labels = np.concatenate((batch_edge_labels, edge_label), axis=None)

    return batched_graphs, torch.tensor(batch_node_labels).float().reshape(-1, 1)# , torch.tensor(batch_edge_labels).float().reshape(-1, 1)


def cli_main():

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--scoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--sim_threshold', type=float, required=True)
    args = parser.parse_args()

    args.gpus = 1

    for random_seed in np.random.randint(low=0, high=10000, size=200):

        L.seed_everything(random_seed, workers=True)

        dgldir = f"{args.outdir}/processed_data/dgl"
        labeldir = f"{args.outdir}/processed_data/label"
        folddir = f"{args.outdir}/fold{args.fold}"

        lines = open(folddir + '/targets.list').readlines()

        targets_train_in_fold = lines[0].split()
        targets_val_in_fold = lines[1].split()
        targets_test_in_fold = lines[2].split()

        print(f"Fold {args.fold}:")

        print(f"Train targets:")
        print(targets_train_in_fold)

        print(f"Validation targets:")
        print(targets_val_in_fold)

        print(f"Test targets:")
        print(targets_test_in_fold)

        if os.path.exists(folddir + '/corr_loss.csv'):
            continue

        ckpt_dir = folddir + '/ckpt/' + str(random_seed)
        os.makedirs(ckpt_dir, exist_ok=True)

        if os.path.exists(ckpt_dir + 'train.done'):
            continue

        batch_size = 512

        train_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_train_in_fold)
        train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=True)
        
        val_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_val_in_fold)
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)

        test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)
        test_loader = DataLoader(test_data,
                                batch_size=1,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)

        node_input_dim = 17
        edge_input_dim = 3

        workdir = f'directed_node_seed_{args.project}_fold{args.fold}_sim{args.sim_threshold}'

        os.makedirs(workdir, exist_ok=True)

        node_input_dim = 17
        edge_input_dim = 3
        num_heads = 4
        num_layer = 4
        dp_rate = 0.3
        hidden_dim = 16
        mlp_dp_rate = 0.3
        layer_norm = True

        # initialise the wandb logger and name your wandb project
        wandb.finish()

        wandb_logger = WandbLogger(project=workdir, save_dir=workdir)

        # add your batch size to the wandb config
        wandb_logger.experiment.config["random_seed"] = random_seed
        wandb_logger.experiment.config["batch_size"] = batch_size
        wandb_logger.experiment.config["node_input_dim"] = node_input_dim
        wandb_logger.experiment.config["edge_input_dim"] = edge_input_dim
        wandb_logger.experiment.config["num_heads"] = num_heads
        wandb_logger.experiment.config["num_layer"] = num_layer
        wandb_logger.experiment.config["dp_rate"] = dp_rate
        wandb_logger.experiment.config["layer_norm"] = layer_norm
        wandb_logger.experiment.config["batch_norm"] = not layer_norm
        wandb_logger.experiment.config["residual"] = True
        wandb_logger.experiment.config["hidden_dim"] = hidden_dim
        wandb_logger.experiment.config["mlp_dp_rate"] = mlp_dp_rate
        wandb_logger.experiment.config["fold"] = args.fold
        wandb_logger.experiment.config["sim_threshold"] = args.sim_threshold
        
        model = Gate(node_input_dim=node_input_dim,
                    edge_input_dim=edge_input_dim,
                    num_heads=num_heads,
                    num_layer=num_layer,
                    dp_rate=dp_rate,
                    layer_norm=layer_norm,
                    batch_norm=not layer_norm,
                    residual=True,
                    hidden_dim=hidden_dim,
                    mlp_dp_rate=mlp_dp_rate,
                    check_pt_dir=ckpt_dir,
                    batch_size=batch_size)

        trainer = L.Trainer(accelerator='gpu',max_epochs=200, logger=wandb_logger)

        wandb_logger.watch(model)

        trainer.fit(model, train_loader, val_loader)

        os.system(f'touch {ckpt_dir}/train.done')
    

if __name__ == '__main__':
    cli_main()
