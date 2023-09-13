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
                      out: str,
                      sim_threshold: float) -> None:
    """Build KNN graph and assign node and edge features. node feature: N * 35, Edge feature: E * 6"""
    if not os.path.exists(subgraph_file):
        raise FileNotFoundError(f'Cannot not find subgraph: {subgraph_file} ')

    # print(f'Processing {filename}')
    scaler = MinMaxScaler()

    subgraph_df = pd.read_csv(subgraph_file, index_col=[0])

    nodes_num = len(subgraph_df)

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
    contact_score_file = score_dir + '/cdpred/' + targetname + '.csv'
    contact_scores_df = pd.read_csv(contact_score_file)
    icps_scores_dict = {k: v for k, v in zip(list(contact_scores_df['model']), list(contact_scores_df['icps']))}
    recall_scores_dict = {k: v for k, v in zip(list(contact_scores_df['model']), list(contact_scores_df['recall']))}

    icps_scores = [icps_scores_dict[model] for model in models]
    recall_scores = [recall_scores_dict[model] for model in models]

    icps_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(icps_scores).reshape(-1, 1))).float()
    recall_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(recall_scores).reshape(-1, 1))).float()

    # To be added: enqa scores
    enqa_score_file = score_dir + '/enqa/' + targetname + '.csv'
    enqa_scores_df = pd.read_csv(enqa_score_file)
    enqa_scores_dict = {k: v for k, v in zip(list(enqa_scores_df['model']), list(enqa_scores_df['score_norm']))}

    enqa_scores = [enqa_scores_dict[model] for model in models]

    # Target too large to run enqa
    if np.sum(np.array(enqa_scores)) == 0:
        enqa_score_feature = torch.tensor(enqa_scores).reshape(-1, 1).float()
    else:
        enqa_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(enqa_scores).reshape(-1, 1))).float()

    # edge features
    # a. global fold similarity between two models
    # b. number of common interfaces
    subgraph_array = np.array(subgraph_df)
    common_interface_csv_file = score_dir + '/edge_features/' + targetname + '.csv'
    common_interface_array = np.array(pd.read_csv(common_interface_csv_file, index_col=[0]))
    
    src_nodes, dst_nodes = [], []
    edge_sim, edge_common_interface = [], []
    for src in range(nodes_num):
        for dst in range(nodes_num):
            if src == dst:
                continue
            if subgraph_array[dst, src] >= sim_threshold:
                src_nodes += [src]
                dst_nodes += [dst]

                # edge_sim += [subgraph_array[dst, src]] # non-symmetric matrix, the similarity score should be noramlized by the target model
                edge_sim += [subgraph_array[src, dst]] # should be normalized by the source model? e.g., source model is larger
                edge_common_interface += [common_interface_array[src, dst]]

    edge_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_sim).reshape(-1, 1))).float()
    edge_common_interface_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_common_interface).reshape(-1, 1))).float()

    # 6. add feature to graph
    graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), num_nodes=nodes_num)
    lap_enc_feature = laplacian_positional_encoding(graph, pos_enc_dim=8)
    update_node_feature(graph, [alphafold_score_feature, 
                                average_sim_score_in_subgraph_feature, average_sim_score_in_full_graph_feature,
                                voro_gnn_score_feature, voro_gnn_pcadscore_feature, voro_dark_score_feature,
                                dproqa_score_feature, icps_score_feature, recall_score_feature, enqa_score_feature, lap_enc_feature])

    edge_sin_pos = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)
    update_edge_feature(graph, [edge_sin_pos, edge_sim_feature, edge_common_interface_feature])

    dgl.save_graphs(filename=os.path.join(out, f'{filename}.dgl'), g_list=graph)
    # print(f'{filename}\nSUCCESS')
    return None


def graph_wrapper(targetname: str, subgraph_file: str, filename: str, score_dir: str, dgl_folder: str, sim_threshold: float):
    build_model_graph(targetname=targetname,
                      subgraph_file=subgraph_file,
                      filename=filename,
                      score_dir=score_dir,
                      out=dgl_folder,
                      sim_threshold=sim_threshold)


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

    np.save(label_folder + '/' + filename + '_node.npy', tmscores)

    # signs = []
    # for i in range(len(models)):
    #     for j in range(len(models)):
    #         if i == j:
    #             continue

    #         if float(tmscore_dict[models[i]]) < float(tmscore_dict[models[j]]):
    #             signs += [0]
    #         else:
    #             signs += [1]

    # np.save(label_folder + '/' + filename + '_edge.npy', signs)


def generate_dgl_and_labels(savedir, targets, datadir, scoredir, sim_threshold):
    # generating graph
    dgl_folder = savedir + '/dgl'
    os.makedirs(dgl_folder, exist_ok=True)

    if not os.path.exists(savedir + '/dgl.done'):
        for target in targets:
            print(f'Generating DGL files for {target}')
            os.makedirs(dgl_folder + '/' + target, exist_ok=True)
            Parallel(n_jobs=-1)(delayed(graph_wrapper)(targetname=target, 
                                                       subgraph_file=datadir + '/' + target + '/' + subgraph_file, 
                                                       filename=target + '_' + subgraph_file.replace('.csv', ''), 
                                                       score_dir=scoredir, 
                                                       dgl_folder=dgl_folder + '/' + target,
                                                       sim_threshold=sim_threshold) 
                                                       for subgraph_file in os.listdir(datadir + '/' + target))
        os.system(f"touch {savedir}/dgl.done")
                    
    # generating labels
    label_folder = savedir + '/label'
    os.makedirs(label_folder, exist_ok=True)

    if not os.path.exists(savedir + '/label.done'):
        for target in targets:
            print(f'Generating label files for {target}')
            os.makedirs(label_folder + '/' + target, exist_ok=True)
            Parallel(n_jobs=10)(delayed(label_wrapper)(targetname=target, 
                                                       subgraph_file=datadir + '/' + target + '/' + subgraph_file, 
                                                       filename=target + '_' + subgraph_file.replace('.csv', ''), 
                                                       score_dir=scoredir, 
                                                       label_folder=label_folder + '/' + target) 
                                                       for subgraph_file in os.listdir(datadir + '/' + target))
        os.system(f"touch {savedir}/label.done")

    return dgl_folder, label_folder        


def cli_main():

    random_seed = 3407

    L.seed_everything(random_seed, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--scoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--sim_threshold', type=float, default=0.0, required=False)
    parser.add_argument('--refdir', type=str, required=True)
    args = parser.parse_args()

    args.gpus = 1

    for sampled_data in os.listdir(args.datadir):
        
        sampled_datadir = args.datadir + '/' + sampled_data

        outdir = args.outdir + '/' + sampled_data

        all_targets = sorted(os.listdir(sampled_datadir))
        
        savedir = outdir + '/processed_data'

        dgl_folder, label_folder = generate_dgl_and_labels(savedir=savedir, targets=all_targets, datadir=sampled_datadir, scoredir=args.scoredir, sim_threshold=args.sim_threshold)

        # kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for i in range(10):
            
            reffolddir = f"{args.refdir}/fold{i}"

            folddir = f"{outdir}/fold{i}"
            os.makedirs(folddir, exist_ok=True)

            print(f"Fold {i}:")

            contents = open(reffolddir + '/targets.list').readlines()

            targets_train_in_ref_fold = contents[0].rstrip('\n').split()
            targets_val_in_ref_fold = contents[1].rstrip('\n').split()
            targets_test_in_ref_fold = contents[2].rstrip('\n').split()

            targets_train_in_fold = []
            for target in targets_train_in_ref_fold:
                if target in all_targets:
                    targets_train_in_fold += [target]

            targets_val_in_fold = []
            for target in targets_val_in_ref_fold:
                if target in all_targets:
                    targets_val_in_fold += [target]

            targets_test_in_fold = []
            for target in targets_test_in_ref_fold:
                if target in all_targets:
                    targets_test_in_fold += [target]   


            print(f"Train targets:")
            print(targets_train_in_fold)

            print(f"Validation targets:")
            print(targets_val_in_fold)

            print(f"Test targets:")
            print(targets_test_in_fold)

            with open(folddir + '/targets.list', 'w') as fw:
                fw.write('\t'.join(targets_train_in_fold) + '\n')
                fw.write('\t'.join(targets_val_in_fold) + '\n')
                fw.write('\t'.join(targets_test_in_fold) + '\n')
    

if __name__ == '__main__':
    cli_main()
