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
from graph_transformer import Gate
import lightning as L
from torch.utils.data import Dataset
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger

def update_node_feature(graph: dgl.DGLGraph, new_node_features: List) -> None:
    """Node feature update helper"""
    for node_feature in new_node_features:
        if not graph.ndata:
            graph.ndata['f'] = node_feature
        else:
            graph.ndata['f'] = torch.cat((graph.ndata['f'], node_feature), dim=1)


def update_edge_feature(graph: dgl.DGLGraph, new_edge_features: List) -> None:
    """Edge feature update helper"""
    for edge_feature in new_edge_features:
        if not graph.edata:
            graph.edata['f'] = edge_feature
        else:
            graph.edata['f'] = torch.cat((graph.edata['f'], edge_feature), dim=1)
    return None


def build_model_graph(targetname: str,
                      subgraph_file: str,
                      filename: str,
                      score_dir: str,
                      out: str) -> None:
    """Build KNN graph and assign node and edge features. node feature: N * 35, Edge feature: E * 6"""
    if not os.path.exists(subgraph_file):
        raise FileNotFoundError(f'Cannot not find subgraph: {subgraph_file} ')

    # print(f'Processing {filename}')
    scaler = MinMaxScaler()

    subgraph_df = pd.read_csv(subgraph_file, index_col=[0])

    nodes_num = len(subgraph_df)

    src_nodes, dst_nodes = [], []
    for i in range(nodes_num):
        for j in range(nodes_num):
            if i == j:
                continue
            src_nodes += [i]
            dst_nodes += [j]

    graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))

    models = subgraph_df.columns

    # node features
    # a. alphafold global plddt score
    alphafold_scores_file = score_dir + '/alphafold/' + targetname + '.csv' 
    alphafold_scores_df = pd.read_csv(alphafold_scores_file)
    alphafold_scores_dict = {k: v for k, v in zip(list(alphafold_scores_df['model']), list(alphafold_scores_df['plddt']))}
    alphafold_scores = [alphafold_scores_dict[model] for model in models]
    
    alphafold_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_scores).reshape(-1, 1))).float()

    # b. average pairwise similarity score
    average_sim_scores = [np.mean(np.array(subgraph_df[model])) for model in models]

    average_sim_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores).reshape(-1, 1))).float()

    # c. voro scores: gnn, gnn_pcadscore, voromqa_dark
    voro_scores_file = score_dir + '/voro_scores/' + targetname + '.csv'
    voro_scores_df = pd.read_csv(voro_scores_file)
    voro_gnn_dict = {k: v for k, v in zip(list(voro_scores_df['model']), list(voro_scores_df['GNN_sum_score']))}
    voro_gnn_pcadscore_dict = {k: v for k, v in zip(list(voro_scores_df['model']), list(voro_scores_df['GNN_pcadscore']))}
    voro_dark_dict = {k: v for k, v in zip(list(voro_scores_df['model']), list(voro_scores_df['voromqa_dark']))}

    voro_gnn_scores = [voro_gnn_dict[model + '.pdb'] for model in models]
    voro_gnn_pcadscores = [voro_gnn_pcadscore_dict[model + '.pdb'] for model in models]
    voro_dark_scores = [voro_dark_dict[model + '.pdb'] for model in models]

    voro_gnn_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(voro_gnn_scores).reshape(-1, 1))).float()
    voro_gnn_pcadscore_feature = torch.tensor(scaler.fit_transform(torch.tensor(voro_gnn_pcadscores).reshape(-1, 1))).float()
    voro_dark_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(voro_dark_scores).reshape(-1, 1))).float()

    # To be added: enqa scores, dproqa scores, contact scores by cdpred


    # edge features
    # a. global fold similarity between two models
    subgraph_array = np.array(subgraph_df)
    edge_sim = []
    for src, dst in zip(src_nodes, dst_nodes):
       edge_sim += [subgraph_array[src, dst]]

    edge_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_sim).reshape(-1, 1))).float()

    # 6. add feature to graph
    update_node_feature(graph, [alphafold_score_feature, average_sim_score_feature,
                                voro_gnn_score_feature, voro_gnn_pcadscore_feature, voro_dark_score_feature])

    update_edge_feature(graph, [edge_sim_feature])

    dgl.save_graphs(filename=os.path.join(out, f'{filename}.dgl'), g_list=graph)
    # print(f'{filename}\nSUCCESS')
    return None


def graph_wrapper(targetname: str, subgraph_file: str, filename: str, score_dir: str, dgl_folder: str):
    build_model_graph(targetname=targetname,
                      subgraph_file=subgraph_file,
                      filename=filename,
                      score_dir=score_dir,
                      out=dgl_folder)


def label_wrapper(targetname: str, subgraph_file: str, filename: str, score_dir: str, label_folder: str):
    if not os.path.exists(subgraph_file):
        raise FileNotFoundError(f'Cannot not find subgraph: {subgraph_file} ')

    # print(f'Processing {filename}')

    subgraph_df = pd.read_csv(subgraph_file, index_col=[0])

    models = subgraph_df.columns

    label_df = pd.read_csv(score_dir + '/label/' + targetname + '.csv')

    tmscore_dict = {k: v for k, v in zip(list(label_df['model']), list(label_df['tmscore']))}

    tmscores = [tmscore_dict[model] for model in models]

    tmscores = np.array(tmscores).reshape(-1, 1)

    np.save(label_folder + '/' + filename + '.npy', tmscores)


class DGLData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str, label_folder: str):
        self.data_list = os.listdir(dgl_folder)
        self.data_path_list = [os.path.join(dgl_folder, i) for i in self.data_list]
        self.label_folder=label_folder

        self.data = []
        self.label = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def _prepare(self):
        for i in range(len(self.data_list)):
            g, tmp = dgl.data.utils.load_graphs(self.data_path_list[i])
            self.data.append(g[0])
            self.label.append(np.load(self.label_folder + '/' + self.data_list[i].replace('.dgl', '.npy')))


def collate(samples):
    """Customer collate function"""
    graphs, labels = zip(*samples)
    batched_graphs = dgl.batch(graphs)
    batch_labels = None
    for label in labels:
        if batch_labels is None:
            batch_labels = copy.deepcopy(label)
        else:
            batch_labels = np.concatenate((batch_labels, label), axis=None)

    return batched_graphs, torch.tensor(batch_labels).float().reshape(-1, 1)


def generate_dgl_and_labels(savedir, targets, datadir, scoredir):
    # generating graph
    dgl_folder = savedir + '/dgl'
    os.makedirs(dgl_folder, exist_ok=True)

    if not os.path.exists(savedir + '/dgl.done'):
        for target in targets:
            print(f'Generating DGL files for {target}')
            Parallel(n_jobs=10)(delayed(graph_wrapper)(targetname=target, 
                                                    subgraph_file=datadir + '/' + target + '/' + subgraph_file, 
                                                    filename=target + '_' + subgraph_file.replace('.csv', ''), 
                                                    score_dir=scoredir, 
                                                    dgl_folder=dgl_folder) 
                                                    for subgraph_file in os.listdir(datadir + '/' + target))
        os.system(f"touch {savedir}/dgl.done")
                    
    # generating labels
    label_folder = savedir + '/label'
    os.makedirs(label_folder, exist_ok=True)

    if not os.path.exists(savedir + '/label.done'):
        for target in targets:
            print(f'Generating label files for {target}')
            Parallel(n_jobs=10)(delayed(label_wrapper)(targetname=target, 
                                                    subgraph_file=datadir + '/' + target + '/' + subgraph_file, 
                                                    filename=target + '_' + subgraph_file.replace('.csv', ''), 
                                                    score_dir=scoredir, 
                                                    label_folder=label_folder) 
                                                    for subgraph_file in os.listdir(datadir + '/' + target))
        os.system(f"touch {savedir}/label.done")

    return dgl_folder, label_folder        


def cli_main():

    random_seed = 1111

    L.seed_everything(random_seed)

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--scoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    args.gpus = 1

    all_targets = sorted(os.listdir(args.datadir))

    targets_train, targets_test = train_test_split(all_targets, test_size=0.1, random_state=42)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(kf.split(targets_train)):
        
        print(f"Fold {i}:")
        targets_train_in_fold = [targets_train[i] for i in train_index]
        targets_val_in_fold = [targets_train[i] for i in val_index]

        folddir = f"{args.outdir}/fold{i}"
        os.makedirs(folddir, exist_ok=True)

        traindir = folddir + '/train'
        os.makedirs(traindir, exist_ok=True)
        train_dgl_folder, train_label_folder = generate_dgl_and_labels(savedir=traindir, targets=targets_train_in_fold, datadir=args.datadir, scoredir=args.scoredir)

        valdir = folddir + '/val'
        os.makedirs(valdir, exist_ok=True)
        val_dgl_folder, val_label_folder = generate_dgl_and_labels(savedir=valdir, targets=targets_val_in_fold, datadir=args.datadir, scoredir=args.scoredir)

        batch_size = 32

        train_data = DGLData(dgl_folder=train_dgl_folder, label_folder=train_label_folder)
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  num_workers=16,
                                  pin_memory=True,
                                  collate_fn=collate,
                                  shuffle=True)
        
        val_data = DGLData(dgl_folder=val_dgl_folder, label_folder=val_label_folder)
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)

        node_input_dim = 5
        edge_input_dim = 1
        num_heads = 4
        num_layer = 4
        dp_rate = 0.3
        layer_norm = False
        batch_norm = True
        residual = True
        hidden_dim = 32
        mlp_dp_rate = 0.3

        # initialise the wandb logger and name your wandb project
        wandb_logger = WandbLogger(project='gate_v1')

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

        model = Gate(node_input_dim=node_input_dim,
                    edge_input_dim=edge_input_dim,
                    num_heads=num_heads,
                    num_layer=num_layer,
                    dp_rate=dp_rate,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    residual=residual,
                    hidden_dim=hidden_dim,
                    mlp_dp_rate=mlp_dp_rate)

        trainer = L.Trainer(accelerator='gpu',max_epochs=1000, logger=wandb_logger)

        trainer.fit(model, train_loader, val_loader)

        os.exit(1)


    

if __name__ == '__main__':
    cli_main()
