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

def laplacian_positional_encoding(g: dgl.DGLGraph, pos_enc_dim: int) -> torch.Tensor:
    """
        Graph positional encoding v/ Laplacian eigenvectors
        :return torch.Tensor (L, pos_enc_dim)
    """

    # Laplacian
    A = g.adjacency_matrix()
    s = torch.sparse_coo_tensor(indices=A.coalesce().indices(),
                                values=A.coalesce().values(),
                                size=A.coalesce().size())
    A = s.to_dense()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.A)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    laplacian_feature = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().reshape(-1, pos_enc_dim)
    return laplacian_feature


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

    lap_enc_feature = laplacian_positional_encoding(graph, pos_enc_dim=8)

    models = subgraph_df.columns

    # node features
    # a. alphafold global plddt score
    alphafold_scores_file = score_dir + '/alphafold/' + targetname + '.csv' 
    alphafold_scores_df = pd.read_csv(alphafold_scores_file)
    alphafold_scores_dict = {k: v for k, v in zip(list(alphafold_scores_df['model']), list(alphafold_scores_df['plddt_norm']))}
    alphafold_scores = [alphafold_scores_dict[model] for model in models]
    
    alphafold_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_scores).reshape(-1, 1))).float()

    # b1. average pairwise similarity score in graph
    # b2. average pairwise similarity score for all models
    average_sim_scores_in_subgraph = [np.mean(np.array(subgraph_df[model])) for model in models]
    average_sim_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_subgraph).reshape(-1, 1))).float()

    fullgraph_file = score_dir + '/pairwise/' + targetname + '.csv'
    full_graph_df = pd.read_csv(fullgraph_file, index_col=[0])
    average_sim_scores_in_full_graph = [np.mean(np.array(full_graph_df[model])) for model in models]
    average_sim_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_full_graph).reshape(-1, 1))).float()

    # c. voro scores: gnn, gnn_pcadscore, voromqa_dark
    voro_scores_file = score_dir + '/voro_scores/' + targetname + '.csv'
    voro_scores_df = pd.read_csv(voro_scores_file)
    voro_gnn_dict = {k: v for k, v in zip(list(voro_scores_df['model']), list(voro_scores_df['GNN_sum_score_norm']))}
    voro_gnn_pcadscore_dict = {k: v for k, v in zip(list(voro_scores_df['model']), list(voro_scores_df['GNN_pcadscore_norm']))}
    voro_dark_dict = {k: v for k, v in zip(list(voro_scores_df['model']), list(voro_scores_df['voromqa_dark_norm']))}

    voro_gnn_scores = [voro_gnn_dict[model] for model in models]
    voro_gnn_pcadscores = [voro_gnn_pcadscore_dict[model] for model in models]
    voro_dark_scores = [voro_dark_dict[model] for model in models]

    voro_gnn_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(voro_gnn_scores).reshape(-1, 1))).float()
    voro_gnn_pcadscore_feature = torch.tensor(scaler.fit_transform(torch.tensor(voro_gnn_pcadscores).reshape(-1, 1))).float()
    voro_dark_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(voro_dark_scores).reshape(-1, 1))).float()

    # d. dproqa scores
    dproqa_score_file = score_dir + '/dproqa/' + targetname + '.csv'
    dproqa_scores_df = pd.read_csv(dproqa_score_file)
    dproqa_scores_dict = {k: v for k, v in zip(list(dproqa_scores_df['model']), list(dproqa_scores_df['DockQ_norm']))}

    dproqa_scores = [dproqa_scores_dict[model] for model in models]
    dproqa_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(dproqa_scores).reshape(-1, 1))).float()

    # e. contact scores by cdpred
    contact_score_file = score_dir + '/contact/' + targetname + '.csv'
    contact_scores_df = pd.read_csv(contact_score_file)
    icps_scores_dict = {k: v for k, v in zip(list(contact_scores_df['model']), list(contact_scores_df['icps']))}
    recall_scores_dict = {k: v for k, v in zip(list(contact_scores_df['model']), list(contact_scores_df['recall']))}

    icps_scores = [icps_scores_dict[model] for model in models]
    recall_scores = [recall_scores_dict[model] for model in models]

    icps_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(icps_scores).reshape(-1, 1))).float()
    recall_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(recall_scores).reshape(-1, 1))).float()

    # To be added: enqa scores

    # edge features
    # a. global fold similarity between two models
    # b. number of common interfaces
    edge_sin_pos = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)

    subgraph_array = np.array(subgraph_df)
    edge_sim = []

    common_interface_csv_file = score_dir + '/edge_features/' + targetname + '.csv'
    common_interface_array = np.array(pd.read_csv(common_interface_csv_file, index_col=[0]))
    edge_common_interface = []

    for src, dst in zip(src_nodes, dst_nodes):
       # edge_sim += [subgraph_array[src, dst]]
       edge_sim += [subgraph_array[dst, src]] # non-symmetric matrix, the similarity score should be noramlized by the target model
       edge_common_interface += [common_interface_array[src, dst]]

    edge_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_sim).reshape(-1, 1))).float()
    edge_common_interface_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_common_interface).reshape(-1, 1))).float()

    # 6. add feature to graph
    update_node_feature(graph, [alphafold_score_feature, 
                                average_sim_score_in_subgraph_feature, average_sim_score_in_full_graph_feature,
                                voro_gnn_score_feature, voro_gnn_pcadscore_feature, voro_dark_score_feature,
                                dproqa_score_feature, icps_score_feature, recall_score_feature, lap_enc_feature])

    update_edge_feature(graph, [edge_sin_pos, edge_sim_feature, edge_common_interface_feature])

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

    qsscore_dict = {k: v for k, v in zip(list(label_df['model']), list(label_df['qscore']))}

    qsscores = [qsscore_dict[model] for model in models]

    qsscores = np.array(qsscores).reshape(-1, 1)

    np.save(label_folder + '/' + filename + '_node.npy', np.concatenate((tmscores, qsscores), axis=0))

    signs = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue

            if float(tmscore_dict[models[i]]) < float(tmscore_dict[models[j]]):
                signs += [0]
            else:
                signs += [1]

    np.save(label_folder + '/' + filename + '_edge.npy', signs)
            

class DGLData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str, label_folder: str, targets):
        self.target_list = targets
        self.dgl_folder = dgl_folder
        self.label_folder = label_folder

        self.data = []
        self.node_label = []
        self.edge_label = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.node_label[idx], self.edge_label[idx]

    def _prepare(self):
        for target in self.target_list:
            target_dgl_folder = self.dgl_folder + '/' + target
            target_label_folder = self.label_folder + '/' + target

            for dgl_file in os.listdir(target_dgl_folder):
                g, tmp = dgl.data.utils.load_graphs(target_dgl_folder + '/' + dgl_file)
                self.data.append(g[0])
                self.node_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_node.npy')))
                self.edge_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_edge.npy')))


def collate(samples):
    """Customer collate function"""
    graphs, node_labels, edge_labels = zip(*samples)
    batched_graphs = dgl.batch(graphs)
    batch_node_labels, batch_edge_labels = None, None
    for node_label in node_labels:
        if batch_node_labels is None:
            batch_node_labels = copy.deepcopy(node_label)
        else:
            batch_node_labels = np.concatenate((batch_node_labels, node_label), axis=None)
    
    for edge_label in edge_labels:
        if batch_edge_labels is None:
            batch_edge_labels = copy.deepcopy(edge_label)
        else:
            batch_edge_labels = np.concatenate((batch_edge_labels, edge_label), axis=None)

    return batched_graphs, torch.tensor(batch_node_labels).float().reshape(-1, 1), torch.tensor(batch_edge_labels).float().reshape(-1, 1)


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

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--scoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--project', type=str, required=True)

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

        batch_size = 256

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

        workdir = f'directed_node_seed_{args.project}_fold{args.fold}'

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
