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
from graph_transformer_v2 import Gate
import lightning as L
from torch.utils.data import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger
import wandb
import scipy.sparse as sp
import torchmetrics
import json

os.environ["WANDB__SERVICE_WAIT"] = "3600"
os.environ["WANDB_API_KEY"] = "e84c57dee287170f97801b73a63280b155507e00"
torch.set_printoptions(profile="full")

class DGLData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str, label_folder: str, targets):
        self.target_list = targets
        self.dgl_folder = dgl_folder
        self.label_folder = label_folder

        self.data_list = []
        self.data = []
        self.node_label = []
        # self.edge_label = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.node_label[idx], self.data_list[idx]#, self.edge_label[idx]

    def _prepare(self):
        for target in self.target_list:
            target_dgl_folder = self.dgl_folder + '/' + target
            target_label_folder = self.label_folder + '/' + target

            for dgl_file in os.listdir(target_dgl_folder):
                g, tmp = dgl.data.utils.load_graphs(target_dgl_folder + '/' + dgl_file)
                self.data.append(g[0])
                self.node_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_node.npy')))
                # self.edge_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_edge.npy')))
                self.data_list.append(target_dgl_folder + '/' + dgl_file)


def collate(samples):
    """Customer collate function"""
    graphs, node_labels, data_paths = zip(*samples)
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

    return batched_graphs, torch.tensor(batch_node_labels).float().reshape(-1, 1), data_paths# , torch.tensor(batch_edge_labels).float().reshape(-1, 1)


def cli_main():

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--scoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--dbdir', type=str, required=True)

    args = parser.parse_args()

    args.gpus = 1

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

    batch_size = 1024

    train_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_train_in_fold)
    val_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_val_in_fold)
    test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)

    random_seed = 3407

    L.seed_everything(random_seed, workers=True)

    projectname = f"{args.project}_fold{args.fold}"

    workdir = f"{args.dbdir}/{args.project}/fold{args.fold}"
    os.makedirs(workdir, exist_ok=True)

    ckpt_root_dir = workdir + '/ckpt/'
    os.makedirs(ckpt_root_dir, exist_ok=True)

    node_input_dim = 20 #18
    edge_input_dim = 4 #3
    layer_norm = True
    residual = True
    batch_norm = not layer_norm

    for num_heads in [4, 8]:
        for num_layer in [3, 4, 5]:
            for dp_rate in [0.2, 0.3, 0.4]:
                for hidden_dim in [16]:
                    for mlp_dp_rate in [0.2, 0.3, 0.4]:
                        for loss_fun in ['mse']:#, 'binary']:
                            for lr in [0.0001, 0.001]:
                                for weight_decay in [0.01]:
                                    
                                    experiment_name = f"{node_input_dim}_" \
                                                      f"{edge_input_dim}_" \
                                                      f"{num_heads}_" \
                                                      f"{num_layer}_" \
                                                      f"{dp_rate}_" \
                                                      f"{layer_norm}_" \
                                                      f"{batch_norm}_" \
                                                      f"{residual}_" \
                                                      f"{hidden_dim}_" \
                                                      f"{mlp_dp_rate}_" \
                                                      f"{loss_fun}_" \
                                                      f"{lr}_" \
                                                      f"{weight_decay}"
                                    
                                    if os.path.exists(f"{ckpt_root_dir}/{experiment_name}.done"):
                                        continue
                            
                                    train_loader = DataLoader(train_data,
                                                            batch_size=batch_size,
                                                            num_workers=24,
                                                            pin_memory=True,
                                                            collate_fn=collate,
                                                            shuffle=True)
                                    
                                    val_loader = DataLoader(val_data,
                                                            batch_size=batch_size,
                                                            num_workers=24,
                                                            pin_memory=True,
                                                            collate_fn=collate,
                                                            shuffle=False)

                                    # initialise the wandb logger and name your wandb project
                                    wandb.finish()

                                    wandb_logger = WandbLogger(project=projectname, save_dir=workdir)

                                    # add your batch size to the wandb config
                                    wandb_logger.experiment.config["random_seed"] = random_seed
                                    wandb_logger.experiment.config["batch_size"] = batch_size
                                    wandb_logger.experiment.config["node_input_dim"] = node_input_dim
                                    wandb_logger.experiment.config["edge_input_dim"] = edge_input_dim
                                    wandb_logger.experiment.config["num_heads"] = num_heads
                                    wandb_logger.experiment.config["num_layer"] = num_layer
                                    wandb_logger.experiment.config["dp_rate"] = dp_rate
                                    wandb_logger.experiment.config["layer_norm"] = layer_norm
                                    wandb_logger.experiment.config["batch_norm"] = batch_norm
                                    wandb_logger.experiment.config["residual"] = residual
                                    wandb_logger.experiment.config["hidden_dim"] = hidden_dim
                                    wandb_logger.experiment.config["mlp_dp_rate"] = mlp_dp_rate
                                    wandb_logger.experiment.config["loss_fun"] = loss_fun
                                    wandb_logger.experiment.config["lr"] = lr
                                    wandb_logger.experiment.config["weight_decay"] = weight_decay
                                    wandb_logger.experiment.config["fold"] = args.fold
                                    
                                    loss_function = None
                                    if loss_fun == 'mse':
                                        loss_function = torchmetrics.MeanSquaredError()
                                    elif loss_fun == 'binary':
                                        loss_function = torch.nn.BCELoss()

                                    if loss_function is None:
                                        continue

                                    ckpt_dir = ckpt_root_dir + '/' + experiment_name
                                    os.makedirs(ckpt_dir, exist_ok=True)

                                    model_dict = {}
                                    with open(ckpt_dir + '/config.json', 'w') as fw:
                                        json.dump(dict(wandb_logger.experiment.config), fw, indent = 4)

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
                                                batch_size=batch_size,
                                                loss_function=loss_function,
                                                learning_rate=lr,
                                                weight_decay=weight_decay)

                                    trainer = L.Trainer(accelerator='gpu',max_epochs=200, logger=wandb_logger, deterministic=True)

                                    wandb_logger.watch(model)

                                    trainer.fit(model, train_loader, val_loader)

                                    os.system(f"touch {ckpt_root_dir}/{experiment_name}.done")
    

if __name__ == '__main__':
    cli_main()
