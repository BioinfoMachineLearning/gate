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
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--ckptdir', type=str, required=True)
    parser.add_argument('--ckptfile', type=str, required=True)

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    ckpts_dict = {}
    for line in open(args.ckptfile):
        line = line.rstrip('\n')
        foldname, ckptname = line.split()
        ckpts_dict[foldname] = ckptname

    for fold in range(10):
        
        dgldir = f"{args.outdir}/processed_data/dgl"
        labeldir = f"{args.outdir}/processed_data/label"
        folddir = f"{args.outdir}/fold{fold}"
        ckpt_dir = f"{args.ckptdir}/fold{fold}/" + ckpts_dict["fold" + str(fold)]
        if len(os.listdir(ckpt_dir)) == 0:
            raise Exception(f"cannot find any check points in {ckpt_dir}")

        if len(os.listdir(ckpt_dir)) >= 3:
            raise Exception(f"multiple check points in {ckpt_dir}")

        for ckptfile in os.listdir(ckpt_dir):
            if ckptfile == "config.json":
                continue
            ckptname = ckptfile
            break

        lines = open(folddir + '/targets.list').readlines()

        targets_test_in_fold = lines[2].split()

        print(f"Fold {fold}:")

        print(f"Test targets:")
        print(targets_test_in_fold)
        
        test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)
        test_loader = DataLoader(test_data,
                                batch_size=1024,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)
        config_file = ckpt_dir + '/config.json'
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
        learning_rate = config_list['lr']
        weight_decay = config_list['weight_decay']
        loss_function = torchmetrics.MeanSquaredError()
        if config_list['loss_fun'] == 'binary':
            loss_function = torch.nn.BCELoss()

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
                    check_pt_dir='',
                    batch_size=512,
                    loss_function=loss_function,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay)

        ckpt_path = ckpt_dir + '/' + ckptname
        print(ckpt_path)
        model = model.load_from_checkpoint(ckpt_path, loss_function=loss_function)

        model = model.to(device)

        model.eval()

        pred_subgraph_scores = {}
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

            start_idx = 0
            for subgraph_path in data_paths:
                subgraph_filename = subgraph_path.split('/')[-1]
                targetname = subgraph_filename.split('_')[0]
                subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
    
                subgraph_df = pd.read_csv(f"{args.datadir}/{targetname}/{subgraph_name.replace('.dgl', '.csv')}", index_col=[0])
                for i, modelname in enumerate(subgraph_df.columns):
                    if modelname not in pred_subgraph_scores:
                        pred_subgraph_scores[modelname] = []
                    pred_subgraph_scores[modelname] += [pred_scores[start_idx + i]]
                start_idx += len(subgraph_df.columns)

        for target in targets_test_in_fold:
            models_for_target = [modelname for modelname in pred_subgraph_scores if modelname.split('TS')[0] == target.replace('o','')]    
            # print(models_for_target)
            ensemble_scores, ensemble_count, std, normalized_std = [], [], [], []
            for modelname in models_for_target:
                mean_score = np.mean(np.array(pred_subgraph_scores[modelname]))
                ensemble_scores += [mean_score]
                ensemble_count += [len(pred_subgraph_scores[modelname])]
                std += [np.std(np.array(pred_subgraph_scores[modelname]))]
                normalized_std += [np.std(np.array(pred_subgraph_scores[modelname])) / mean_score]
            pd.DataFrame({'model': models_for_target, 'score': ensemble_scores, 
                          'sample_count': ensemble_count, 'std': std, "std_norm": normalized_std}).to_csv(folddir + '/' + target + '.csv')

    

if __name__ == '__main__':
    cli_main()
