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
from graph_transformer_v3 import Gate
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
    parser.add_argument('--ckptfiledir', type=str, required=True)

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    savedir = args.outdir + '/predictions/'
    os.makedirs(savedir, exist_ok=True)

    for fold in range(10):
    #for fold in [5]:    
        msefile = f"{args.ckptfiledir}/fold{fold}/mse.csv"
        df = pd.read_csv(msefile)
        ckptnames = list(df['ckptdir'])

        dgldir = f"{args.outdir}/processed_data/dgl"
        labeldir = f"{args.outdir}/processed_data/label"
        folddir = f"{args.outdir}/fold{fold}"

        lines = open(folddir + '/targets.list').readlines()

        targets_test_in_fold = lines[2].split()

        print(f"Fold {fold}:")

        print(f"Test targets:")
        print(targets_test_in_fold)
        
        savefolddir= savedir + f'/fold{fold}'
        os.makedirs(savefolddir, exist_ok=True)

        ensemble_dict = {}
        for i in range(len(df)):
            ckptname = df.loc[i, 'ckptdir']
            valid_target_mean_ranking_loss = df.loc[i, 'val_target_mean_ranking_loss']
            valid_target_median_ranking_loss = df.loc[i, 'val_target_median_ranking_loss']
            valid_target_mean_mse = df.loc[i, 'val_target_mean_mse']
            valid_target_median_mse = df.loc[i, 'val_target_median_mse']

            if valid_target_mean_ranking_loss == valid_target_median_ranking_loss:
                if valid_target_mean_mse < valid_target_median_mse:
                    ensemble_dict[ckptname] = 'mean'
                else:
                    ensemble_dict[ckptname] = 'median'
            elif valid_target_mean_ranking_loss < valid_target_median_ranking_loss:
                ensemble_dict[ckptname] = 'mean'
            else:
                ensemble_dict[ckptname] = 'median'

        for ckptname in ckptnames:

            savefoldckptdir = savefolddir + '/' + ckptname
            os.makedirs(savefoldckptdir, exist_ok=True)

            bRun = False
            for target in targets_test_in_fold:
                if not os.path.exists(savefoldckptdir + '/' + target + '.csv'):
                    bRun = True
                    break
            
            if not bRun:
                continue

            ckpt_dir = f"{args.ckptdir}/fold{fold}/ckpt/" + ckptname
            if len(os.listdir(ckpt_dir)) == 0:
                raise Exception(f"cannot find any check points in {ckpt_dir}")

            if len([ckptfile for ckptfile in os.listdir(ckpt_dir) if ckptfile.find('.ckpt') >= 0]) >= 3:
                raise Exception(f"multiple check points in {ckpt_dir}")
            
            for ckptfile in os.listdir(ckpt_dir):
                if ckptfile.find('.ckpt') < 0:
                    continue
                ckpt_path = ckpt_dir + '/' + ckptfile
                break

            print(ckpt_path)
               
            config_file = ckpt_dir + '/config.json'

            if os.path.exists(config_file):
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
                opt = config_list['opt']
                learning_rate = config_list['lr']
                weight_decay = config_list['weight_decay']
                loss_function = torchmetrics.MeanSquaredError()
                batch_size = config_list['batch_size']
                if config_list['loss_fun'] == 'binary':
                    loss_function = torch.nn.BCELoss()
                pairwise_loss_function = torchmetrics.MeanSquaredError()
                pairwise_loss_weight = config_list['pairwise_loss_weight']

            else:
                raise Exception(f"Cannot find the config file: {config_file}")

            test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)
            test_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                num_workers=8,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)

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
                        pairwise_loss_function=pairwise_loss_function,
                        pairwise_loss_weight=pairwise_loss_weight,
                        opt=opt,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay)
            
            model = model.load_from_checkpoint(ckpt_path, loss_function=loss_function)

            model = model.to(device)

            model.eval()

            target_pred_subgraph_scores = {}
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
                    
                    if targetname not in target_pred_subgraph_scores:
                        target_pred_subgraph_scores[targetname] = {}

                    subgraph_df = pd.read_csv(f"{args.datadir}/{targetname}/{subgraph_name.replace('.dgl', '.csv')}", index_col=[0])
                    for i, modelname in enumerate(subgraph_df.columns):
                        if modelname not in target_pred_subgraph_scores[targetname]:
                            target_pred_subgraph_scores[targetname][modelname] = []
                        target_pred_subgraph_scores[targetname][modelname] += [pred_scores[start_idx + i]]
                    start_idx += len(subgraph_df.columns)

            for target in targets_test_in_fold:
                ensemble_scores, ensemble_count, std, normalized_std = [], [], [], []
                for modelname in target_pred_subgraph_scores[target]:
                    target_pred_outdir = folddir + '/' + target
                    os.makedirs(target_pred_outdir, exist_ok=True)
                    # with open(target_pred_outdir + '/' + modelname, 'w') as fw:
                    #     for pred_score in target_pred_subgraph_scores[target][modelname]:
                    #         fw.write(str(pred_score) + '\n')
                    mean_score = np.mean(np.array(target_pred_subgraph_scores[target][modelname]))
                    # ensemble_scores += [mean_score]
                    median_score = np.median(np.array(target_pred_subgraph_scores[target][modelname]))
                    #ensemble_scores += [mean_score]
                    if ensemble_dict[ckptname] == "mean":
                        ensemble_scores += [mean_score]
                    else:
                        ensemble_scores += [median_score]
                        
                    ensemble_count += [len(target_pred_subgraph_scores[target][modelname])]
                    std += [np.std(np.array(target_pred_subgraph_scores[target][modelname]))]
                    normalized_std += [np.std(np.array(target_pred_subgraph_scores[target][modelname])) / mean_score]
                pd.DataFrame({'model': list(target_pred_subgraph_scores[target].keys()), 'score': ensemble_scores, 
                            'sample_count': ensemble_count, 'std': std, "std_norm": normalized_std}).to_csv(savefoldckptdir + '/' + target + '.csv')

    

if __name__ == '__main__':
    cli_main()
