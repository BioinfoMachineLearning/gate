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
    parser.add_argument('--ckptdir', type=str, required=True)

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    for fold in range(10):
        
        dgldir = f"{args.outdir}/processed_data/dgl"
        labeldir = f"{args.outdir}/processed_data/label"
        folddir = f"{args.outdir}/fold{fold}"
        ckpt_dir = f"{args.ckptdir}/fold{fold}"

        if len(os.listdir(ckpt_dir)) == 0:
            continue

        lines = open(folddir + '/targets.list').readlines()

        targets_test_in_fold = lines[2].split()

        print(f"Fold {fold}:")

        print(f"Test targets:")
        print(targets_test_in_fold)
        
        if os.path.exists(folddir + '/corr_loss.csv'):
            continue

        test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)
        test_loader = DataLoader(test_data,
                                batch_size=1,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)

        node_input_dim = 18
        edge_input_dim = 3
        num_heads = 4
        num_layer = 3
        if fold == 2:
            num_layer = 4

        dp_rate = 0
        hidden_dim = 16
        mlp_dp_rate = 0
        layer_norm = True

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
                    batch_size=1,
                    loss_function=torch.nn.BCELoss(),
                    learning_rate=0.1,
                    weight_decay=0.01)

        ckpt_path = ckpt_dir + '/' + os.listdir(ckpt_dir)[0]

        model = model.load_from_checkpoint(ckpt_path, loss_function=torch.nn.BCELoss())

        model = model.to(device)

        model.eval()

        pred_subgraph_scores = {}
        for idx, batch_graphs in enumerate(test_loader):
            subgraph = batch_graphs[0]
            batch_x = subgraph.ndata['f'].to(torch.float)
            batch_e = subgraph.edata['f'].to(torch.float)
            subgraph = subgraph.to(device)
            batch_x = batch_x.to(device)
            batch_e = batch_e.to(device)
            batch_scores = model.forward(subgraph, batch_x, batch_e)

            subgraph_paths = test_data.data_list[idx].split('/')
            subgraph_filename = subgraph_paths[len(subgraph_paths)-1]
            targetname = subgraph_filename.split('_')[0]
            subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
    
            subgraph_df = pd.read_csv(f"{args.datadir}/{targetname}/{subgraph_name.replace('.dgl', '.csv')}", index_col=[0])
            pred_scores = batch_scores.cpu().data.numpy().squeeze(1)
            for i, modelname in enumerate(subgraph_df.columns):
                if modelname not in pred_subgraph_scores:
                    pred_subgraph_scores[modelname] = []
                pred_subgraph_scores[modelname] += [pred_scores[i]]

        # print(pred_subgraph_scores)
        fw = open(folddir + '/corr_loss.csv', 'w')

        for target in targets_test_in_fold:
            models_for_target = [modelname for modelname in pred_subgraph_scores if modelname.split('TS')[0] == target.replace('o','')]    
            # print(models_for_target)
            ensemble_scores = []
            for modelname in models_for_target:
                mean_score = np.mean(np.array(pred_subgraph_scores[modelname]))
                ensemble_scores += [mean_score]
            pd.DataFrame({'model': models_for_target, 'score': ensemble_scores}).to_csv(folddir + '/' + target + '.csv')
        
            # native_score_file = args.scoredir + '/label/' + target + '.csv'
            # native_df = pd.read_csv(native_score_file)

            # native_scores_dict = {}
            # for i in range(len(native_df)):
            #     native_scores_dict[native_df.loc[i, 'model']] = float(native_df.loc[i,'tmscore'])

            # corr = pearsonr(np.array(ensemble_scores), np.array(native_df['tmscore']))[0]
            
            # pred_df = pd.read_csv(folddir + '/' + target + '.csv')
            # pred_df = pred_df.sort_values(by=['score'], ascending=False)
            # pred_df.reset_index(inplace=True)

            # top1_model = pred_df.loc[0, 'model']

            # ranking_loss = float(np.max(np.array(native_df['tmscore']))) - float(native_scores_dict[top1_model])

            # print(f"Target\tcorr\tloss")
            # print(f"{target}\t{corr}\t{ranking_loss}")

            # fw.write(f"Target\tcorr\tloss\n")
            # fw.write(f"{target}\t{corr}\t{ranking_loss}\n")

            # pairwise_df = pd.read_csv(args.scoredir + '/pairwise/' + target + '.csv')
            # average_pairwise_scores = []
            # for modelname in models_for_target:
            #     average_pairwise_scores += []

            
            # corr_pair = pearsonr(np.array(pairwise_df['']), np.array(native_df['tmscore']))[0]
        
        fw.close()
    

if __name__ == '__main__':
    cli_main()
