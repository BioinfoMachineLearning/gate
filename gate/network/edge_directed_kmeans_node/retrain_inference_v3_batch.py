"""
@ Description: Protein to DGL graph with node and edge features
"""

import os, copy, time
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
import scipy.sparse as sp
import json
import torchmetrics
import torch.nn as nn

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
    node_counts = []
    for graph in graphs:
        node_counts += [graph.number_of_nodes()]

    batched_graphs = dgl.batch(graphs)
    batch_node_labels, batch_edge_labels = None, None
    for node_label in node_labels:
        if batch_node_labels is None:
            batch_node_labels = copy.deepcopy(node_label)
        else:
            batch_node_labels = np.concatenate((batch_node_labels, node_label), axis=None)
    
    return batched_graphs, torch.tensor(batch_node_labels).float().reshape(-1, 1), data_paths, node_counts

def read_subgraph_columns(datadir, targets):
    subgraph_columns_dict = {}
    for targetname in os.listdir(datadir):
        if targetname not in targets:
            continue
        for subgraph in os.listdir(datadir + '/' + targetname):
            df = pd.read_csv(f"{datadir}/{targetname}/{subgraph}", index_col=[0])
            subgraph_columns_dict[f"{targetname}_{subgraph.replace('.csv', '')}"] = df.columns
    return subgraph_columns_dict

def read_native_dfs(labeldir, targets):
    native_dfs_dict = {}
    for infile in os.listdir(labeldir):
        targetname = infile.replace('.csv', '')
        if targetname in targets:
            df = pd.read_csv(labeldir + '/' + infile)
            native_dfs_dict[targetname] = df
    return native_dfs_dict

    
def cli_main():

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--ckptdir', type=str, required=True)
    parser.add_argument('--wandbdir', type=str, required=True)
    parser.add_argument('--workdir', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--labeldir', type=str, required=True)

    args = parser.parse_args()

    # df = pd.read_csv(args.ckptfile)
    
    random_seed = 3407

    L.seed_everything(random_seed, workers=True)

    for fold in range(8, 10):
            
        df = pd.read_csv(f"{args.wandbdir}/fold{fold}/mse.csv")
        
        dgldir = f"{args.outdir}/processed_data/dgl"
        labeldir = f"{args.outdir}/processed_data/label"
        folddir = f"{args.outdir}/fold{fold}"

        # continue
        lines = open(folddir + '/targets.list').readlines()

        targets_train_in_fold = lines[0].split()
        targets_val_in_fold = lines[1].split()
        targets_test_in_fold = lines[2].split()

        print(f"Fold {fold}:")

        print(f"Train targets:")
        print(targets_train_in_fold)

        print(f"Validation targets:")
        print(targets_val_in_fold)

        print(f"Test targets:")
        print(targets_test_in_fold)

        # batch_size = 512

        start = time.time()
        # train_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_train_in_fold)
        val_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_val_in_fold)
        # test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)

        load_targets = []
        load_targets += targets_val_in_fold

        subgraph_columns_dict = read_subgraph_columns(args.datadir, load_targets)
        native_dfs_dict = read_native_dfs(args.labeldir, load_targets)
        end = time.time()
        
        print(f"Loading time: {end-start}s")

        for ckpt_name in list(df['ckptdir']):
            ckpt_dir = f"{args.ckptdir}/fold{fold}/ckpt/" + ckpt_name
            if len([ckptfile for ckptfile in os.listdir(ckpt_dir) if ckptfile.find('.ckpt') >= 0]) >= 3:
                raise Exception(f"multiple check points in {ckpt_dir}")

            for ckptfile in os.listdir(ckpt_dir):
                if ckptfile.find('.ckpt') < 0:
                    continue
                ckptname = ckptfile
                break

            config_file = ckpt_dir + '/config.json'

            if not os.path.exists(config_file):
                raise Exception(f"Cannot find the config file: {config_file}")

            with open(config_file) as f:
                config_list = json.load(f)
            node_input_dim = config_list['node_input_dim']
            edge_input_dim = config_list['edge_input_dim']
            num_heads = config_list['num_heads']
            num_layer = config_list['num_layer']
            dp_rate = config_list['dp_rate']
            hidden_dim = config_list['hidden_dim']
            mlp_dp_rate = config_list['mlp_dp_rate']
            layer_norm = config_list['layer_norm']
            batch_norm = not layer_norm
            residual = True
            learning_rate = config_list['lr']
            weight_decay = config_list['weight_decay']
            loss_fun = config_list['loss_fun']
            batch_size = config_list['batch_size']

            workdir = os.path.join(args.workdir, args.project, f"fold{fold}")
            ckptdir = f"{workdir}/ckpt"
            os.makedirs(ckptdir, exist_ok=True)
        
            # train_loader = DataLoader(train_data,
            #                         batch_size=batch_size,
            #                         num_workers=16,
            #                         pin_memory=True,
            #                         collate_fn=collate,
            #                         shuffle=True)
            
            val_loader = DataLoader(val_data,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    pin_memory=True,
                                    collate_fn=collate,
                                    shuffle=False)

            # initialise the wandb logger and name your wandb project
            wandb.finish()

            wandb_logger = WandbLogger(project=args.project + f"_fold{fold}", save_dir=workdir)

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
            wandb_logger.experiment.config["lr"] = learning_rate
            wandb_logger.experiment.config["weight_decay"] = weight_decay
            wandb_logger.experiment.config["fold"] = fold
            
            loss_function = None
            if loss_fun == 'mse':
                loss_function = torchmetrics.MeanSquaredError()
            elif loss_fun == 'binary':
                loss_function = torch.nn.BCELoss()

            if loss_function is None:
                continue

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
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        train_targets=targets_train_in_fold,
                        valid_targets=targets_val_in_fold,
                        subgraph_columns_dict=subgraph_columns_dict,
                        native_dfs_dict=native_dfs_dict,
                        log_train_mse=False,
                        log_val_mse=True)

            trainer = L.Trainer(accelerator='gpu',max_epochs=200, logger=wandb_logger, deterministic=True)

            # wandb_logger.watch(model)

            # trainer.fit(model, train_loader, val_loader)
            ckpt_path = ckpt_dir + '/' + ckptname
            model = model.load_from_checkpoint(ckpt_path, loss_function=loss_function, 
                                                valid_targets=targets_val_in_fold, subgraph_columns_dict=subgraph_columns_dict, 
                                                native_dfs_dict=native_dfs_dict, log_val_mse=True)
            trainer.test(model, dataloaders=val_loader)

    

if __name__ == '__main__':
    cli_main()
