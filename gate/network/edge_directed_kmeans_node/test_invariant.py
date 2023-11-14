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
import scipy.sparse as sp
import json
import torchmetrics

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
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    dgldir = f"{args.outdir}/processed_data/dgl"
    labeldir = f"{args.outdir}/processed_data/label"
    
    targets_test_in_fold = ['T1181o']

    print(f"Test targets:")
    print(targets_test_in_fold)
    
    test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)
    test_loader = DataLoader(test_data,
                            batch_size=1,
                            num_workers=32,
                            pin_memory=True,
                            collate_fn=collate,
                            shuffle=False)

    model = Gate(node_input_dim=22,#14,
                edge_input_dim=5,#4,
                num_heads=4,
                num_layer=2,
                dp_rate=0,
                layer_norm=True,
                batch_norm=False,
                residual=True,
                hidden_dim=16,
                mlp_dp_rate=0,
                check_pt_dir='',
                batch_size=1,
                loss_function=torchmetrics.MeanSquaredError(),
                learning_rate=0.1,
                weight_decay=0.1)
    
    model = model.to(device)

    model.eval()

    for idx, (batch_graphs, labels, data_paths) in enumerate(test_loader):
        #print(data_paths)
        #subgraph = batch_graphs[0]
        batch_x = batch_graphs.ndata['f'].to(torch.float)
        batch_e = batch_graphs.edata['f'].to(torch.float)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_x.to(device)
        batch_e = batch_e.to(device)
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        pred_scores = batch_scores.cpu().data.numpy().squeeze(1)
        print(pred_scores)
        print(np.sum(pred_scores))

if __name__ == '__main__':
    cli_main()
