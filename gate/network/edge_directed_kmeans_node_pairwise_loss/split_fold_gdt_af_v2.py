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
from scipy.stats.stats import pearsonr
import scipy.sparse as sp
import random

def read_qa_txt_as_df(targetname, infile):
    models = []
    scores = []
    for line in open(infile):
        line = line.rstrip('\n')
        # print(line)
        if len(line) == 0:
            continue
        contents = line.split()
        # if contents[0] == "QMODE":
        #     if float(contents[1]) == 2:
        #         return None

        if contents[0] == "PFRMAT" or contents[0] == "TARGET" or contents[0] == "MODEL" or contents[0] == "QMODE" or \
                contents[0] == "END" or contents[0] == "REMARK":
            continue

        model = contents[0]

        if model.find('/') >= 0:
            model = os.path.basename(model)
            # print(model)

        score = contents[1]

        models += [model]
        scores += [float(score)]

    df = pd.DataFrame({'model': models, 'score': scores})
    df = df.sort_values(by=['score'], ascending=False)
    df.reset_index(inplace=True)
    # print(df)
    return df

def read_deeprank3_features(targetname, file_list, models):
    #print(file_list)
    scaler = MinMaxScaler()
    deeprank3_features = []
    for qafile in file_list:
        # print(f"reading {qafile}")
        try:
            scores_df = read_qa_txt_as_df(targetname, qafile)
            scores_dict = {k: v for k, v in zip(list(scores_df['model']), list(scores_df['score']))}            
            scores = [scores_dict[model] for model in models]
            scores_feature = torch.tensor(scaler.fit_transform(torch.tensor(scores).reshape(-1, 1))).float()
            deeprank3_features += [scores_feature]
        except Exception as e:
            print(f"Error in reading {qafile}")
            print(e)
            os.exit(1)
    return deeprank3_features

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
                      sim_threshold: float,
                      auto_sim_threshold: bool,
                      auto_sim_threshold_global: bool) -> None:
    """Build KNN graph and assign node and edge features. node feature: N * 35, Edge feature: E * 6"""
    if not os.path.exists(subgraph_file):
        raise FileNotFoundError(f'Cannot not find subgraph: {subgraph_file} ')

    # print(f'Processing {filename}')
    scaler = MinMaxScaler()

    subgraph_gdtscore_df = pd.read_csv(subgraph_file, index_col=[0])
    subgraph_tmscore_df = pd.read_csv(subgraph_file.replace('gdtscore', 'tmscore'), index_col=[0])
    subgraph_cad_score_df = pd.read_csv(subgraph_file.replace('gdtscore', 'cad_score'), index_col=[0])
    subgraph_lddt_df = pd.read_csv(subgraph_file.replace('gdtscore', 'lddt'), index_col=[0])

    nodes_num = len(subgraph_gdtscore_df)

    models = subgraph_gdtscore_df.columns

    # node features
    # a. alphafold global plddt score
    alphafold_scores_file = score_dir + '/alphafold/' + targetname + '.csv' 
    alphafold_scores_df = pd.read_csv(alphafold_scores_file)
    alphafold_scores_dict = {k: v for k, v in zip(list(alphafold_scores_df['model']), list(alphafold_scores_df['plddt']))}
    alphafold_plddt_scores = [alphafold_scores_dict[model] for model in models]
    
    alphafold_plddt_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_plddt_scores).reshape(-1, 1))).float()
    
    # gdtscore
    # b1. average pairwise similarity score in graph
    # b2. average pairwise similarity score for all models

    average_sim_scores_in_subgraph = [np.mean(np.array(subgraph_gdtscore_df[model])) for model in models]
    average_sim_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_subgraph).reshape(-1, 1))).float()

    fullgraph_gdtscore_file = score_dir + '/pairwise/' + targetname + '_gdtscore.csv'
    full_gdtscore_graph_df = pd.read_csv(fullgraph_gdtscore_file, index_col=[0])
    average_sim_scores_in_full_graph = [np.mean(np.array(full_gdtscore_graph_df[model])) for model in models]
    average_sim_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_full_graph).reshape(-1, 1))).float()

    # tmscore
    average_sim_tmscore_in_subgraph = [np.mean(np.array(subgraph_tmscore_df[model])) for model in models]
    average_sim_tmscore_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_tmscore_in_subgraph).reshape(-1, 1))).float()

    fullgraph_tmscore_file = score_dir + '/pairwise/' + targetname + '_tmscore.csv'
    full_tmscore_graph_df = pd.read_csv(fullgraph_tmscore_file, index_col=[0])
    average_sim_tmscore_in_full_graph = [np.mean(np.array(full_tmscore_graph_df[model])) for model in models]
    average_sim_tmscore_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_tmscore_in_full_graph).reshape(-1, 1))).float()

    # cad-score
    average_sim_cad_scores_in_subgraph = [np.mean(np.array(subgraph_cad_score_df[model])) for model in models]
    average_sim_cad_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_cad_scores_in_subgraph).reshape(-1, 1))).float()

    fullgraph_cad_score_file = score_dir + '/interface_pairwise/' + targetname + '_cad_score.csv'
    full_cad_score_graph_df = pd.read_csv(fullgraph_cad_score_file, index_col=[0])
    average_sim_cad_scores_in_full_graph = [np.mean(np.array(full_cad_score_graph_df[model])) for model in models]
    average_sim_cad_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_cad_scores_in_full_graph).reshape(-1, 1))).float()

    # lddt
    average_sim_lddt_in_subgraph = [np.mean(np.array(subgraph_lddt_df[model])) for model in models]
    average_sim_lddt_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_lddt_in_subgraph).reshape(-1, 1))).float()

    fullgraph_lddt_file = score_dir + '/interface_pairwise/' + targetname + '_lddt.csv'
    full_lddt_graph_df = pd.read_csv(fullgraph_lddt_file, index_col=[0])
    average_sim_lddt_in_full_graph = [np.mean(np.array(full_lddt_graph_df[model])) for model in models]
    average_sim_lddt_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_lddt_in_full_graph).reshape(-1, 1))).float()

    # To be added: enqa scores
    enqa_score_file = score_dir + '/enqa/' + targetname + '.csv'
    enqa_scores_df = pd.read_csv(enqa_score_file)
    enqa_scores_dict = {k: v for k, v in zip(list(enqa_scores_df['model']), list(enqa_scores_df['score']))}

    enqa_scores = [enqa_scores_dict[model] for model in models]

    # Target too large to run enqa
    if np.sum(np.array(enqa_scores)) == 0:
        enqa_score_feature = torch.tensor(enqa_scores).reshape(-1, 1).float()
    else:
        enqa_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(enqa_scores).reshape(-1, 1))).float()

    # d. gcpnet scores
    gcpnet_score_file = score_dir + '/gcpnet_ema_pdb/' + targetname + '/' + targetname + '_esm_plddt.csv'
    gcpnet_scores_df = pd.read_csv(gcpnet_score_file)
    gcpnet_scores_dict = {k: v for k, v in zip(list(gcpnet_scores_df['model']), list(gcpnet_scores_df['score']))}

    gcpnet_scores = [gcpnet_scores_dict[model] for model in models]
    gcpnet_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(gcpnet_scores).reshape(-1, 1))).float()


    # DeepRank3 features

    ranking_files = []

    ranking_files += [score_dir + '/deeprank3/' + targetname + '/DeepRank3_Cluster.txt']
    ranking_files += [score_dir + '/deeprank3/' + targetname + '/DeepRank3_SingleQA.txt']
    ranking_files += [score_dir + '/deeprank3/' + targetname + '/DeepRank3_SingleQA_lite.txt']

    deeprank3_feature_dir = score_dir + '/deeprank3/' + targetname + '/ALL_14_scores'
    deeprank3_feature_names = ['feature_DeepQA',
                                'feature_dist_gist',
                                'feature_dist_orb_num',
                                'feature_dist_pearson',
                                'feature_dist_phash',
                                'feature_dist_precl2_long',
                                'feature_dist_precl2',
                                'feature_dist_psnr',
                                'feature_dist_recall_long',
                                'feature_dist_recall',
                                'feature_dist_rmse',
                                'feature_dist_ssim',
                                'feature_dncon4_long-range',
                                'feature_dncon4_medium-range',
                                'feature_dncon4_short-range',
                                'feature_dope',
                                'feature_OPUS',
                                'feature_pairwiseScore',
                                'feature_pcons',
                                'feature_proq2',
                                'feature_proq3_highres',
                                'feature_proq3_lowres',
                                'feature_proq3',
                                'feature_RF_SRS',
                                'feature_RWplus',
                                'feature_SBROD',
                                'feature_voronota',
                                'modfoldclust2']

    for deeprank3_feature_name in deeprank3_feature_names:
        ranking_files += [os.path.join(deeprank3_feature_dir, deeprank3_feature_name + '.' + targetname)]

    deeprank3_features = read_deeprank3_features(targetname, ranking_files, models)

    # edge features
    # a. global fold similarity between two models
    # b. number of common interfaces
    subgraph_gdtscore_array = np.array(subgraph_gdtscore_df)
    subgraph_tmscore_array = np.array(subgraph_tmscore_df)
    subgraph_cad_score_array = np.array(subgraph_cad_score_df)
    subgraph_lddt_array = np.array(subgraph_lddt_df)

    if auto_sim_threshold:
        if auto_sim_threshold_global:
            sim_threshold = np.mean(np.array(full_gdtscore_graph_df))
        else:
            sim_threshold = np.mean(np.array(subgraph_gdtscore_df))

    # print(sim_threshold)
    src_nodes, dst_nodes = [], []
    edge_gdtscore_sim, edge_tmscore_sim, edge_cad_score_sim, edge_lddt_sim = [], [], [], []
    for src in range(nodes_num):
        for dst in range(nodes_num):
            if src == dst:
                continue
            if subgraph_gdtscore_array[src, dst] >= sim_threshold:
                src_nodes += [src]
                dst_nodes += [dst]

                # edge_sim += [subgraph_mmalign_array[dst, src]] # non-symmetric matrix, the similarity score should be noramlized by the target model
                edge_gdtscore_sim += [subgraph_gdtscore_array[src, dst]] # should be normalized by the source model? e.g., source model is larger
                edge_tmscore_sim += [subgraph_tmscore_array[src, dst]]
                edge_cad_score_sim += [subgraph_cad_score_array[src, dst]]
                edge_lddt_sim += [subgraph_lddt_array[src, dst]]

    if len(edge_gdtscore_sim) > 0:
        # 6. add feature to graph
        graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), num_nodes=nodes_num)
        # lap_enc_feature = laplacian_positional_encoding(graph, pos_enc_dim=8)
        update_node_feature(graph, [alphafold_plddt_score_feature, 
                                    average_sim_score_in_subgraph_feature, average_sim_score_in_full_graph_feature,
                                    average_sim_tmscore_in_subgraph_feature, average_sim_tmscore_in_full_graph_feature,
                                    average_sim_cad_score_in_subgraph_feature, average_sim_cad_score_in_full_graph_feature,
                                    average_sim_lddt_in_subgraph_feature, average_sim_lddt_in_full_graph_feature,
                                    enqa_score_feature, gcpnet_score_feature] + deeprank3_features)

        edge_gdtscore_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_gdtscore_sim).reshape(-1, 1))).float()
        edge_tmscore_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_tmscore_sim).reshape(-1, 1))).float()
        edge_cad_score_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_cad_score_sim).reshape(-1, 1))).float()
        edge_lddt_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_lddt_sim).reshape(-1, 1))).float()

        update_edge_feature(graph, [edge_gdtscore_sim_feature, edge_tmscore_sim_feature,
                                    edge_cad_score_sim_feature, edge_lddt_sim_feature])

    else:
        graph = dgl.DGLGraph()
        graph.add_nodes(nodes_num)
        update_node_feature(graph, [alphafold_plddt_score_feature, 
                                    average_sim_score_in_subgraph_feature, average_sim_score_in_full_graph_feature,
                                    average_sim_tmscore_in_subgraph_feature, average_sim_tmscore_in_full_graph_feature,
                                    average_sim_cad_score_in_subgraph_feature, average_sim_cad_score_in_full_graph_feature,
                                    average_sim_lddt_in_subgraph_feature, average_sim_lddt_in_full_graph_feature,
                                    enqa_score_feature, gcpnet_score_feature] + deeprank3_features)

    dgl.save_graphs(filename=os.path.join(out, f'{filename}.dgl'), g_list=graph)
    # print(f'{filename}\nSUCCESS')
    return None


def graph_wrapper(targetname: str, subgraph_file: str, filename: str, score_dir: str, dgl_folder: str, 
                  sim_threshold: float, auto_sim_threshold: bool, auto_sim_threshold_global: bool):
    build_model_graph(targetname=targetname,
                      subgraph_file=subgraph_file,
                      filename=filename,
                      score_dir=score_dir,
                      out=dgl_folder,
                      sim_threshold=sim_threshold,
                      auto_sim_threshold=auto_sim_threshold,
                      auto_sim_threshold_global=auto_sim_threshold_global)


def label_wrapper(targetname: str, subgraph_file: str, filename: str, score_dir: str, label_folder: str):
    if not os.path.exists(subgraph_file):
        raise FileNotFoundError(f'Cannot not find subgraph: {subgraph_file} ')

    # print(f'Processing {filename}')

    subgraph_gdtscore_df = pd.read_csv(subgraph_file, index_col=[0])

    models = subgraph_gdtscore_df.columns

    label_df = pd.read_csv(score_dir + '/label/' + targetname + '.csv')

    tmscore_dict = {k: v for k, v in zip(list(label_df['model']), list(label_df['gdtscore']))}

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


def generate_dgl_and_labels(savedir, targets, datadir, scoredir, sim_threshold, auto_sim_threshold, auto_sim_threshold_global):
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
                                                       sim_threshold=sim_threshold,
                                                       auto_sim_threshold=auto_sim_threshold,
                                                       auto_sim_threshold_global=auto_sim_threshold_global) 
                                                       for subgraph_file in os.listdir(datadir + '/' + target) if subgraph_file.find('gdtscore') > 0)
        os.system(f"touch {savedir}/dgl.done")
                    
    # generating labels
    label_folder = savedir + '/label'
    os.makedirs(label_folder, exist_ok=True)

    if not os.path.exists(savedir + '/label.done'):
        for target in targets:
            print(f'Generating label files for {target}')
            os.makedirs(label_folder + '/' + target, exist_ok=True)
            Parallel(n_jobs=-1)(delayed(label_wrapper)(targetname=target, 
                                                       subgraph_file=datadir + '/' + target + '/' + subgraph_file, 
                                                       filename=target + '_' + subgraph_file.replace('.csv', ''), 
                                                       score_dir=scoredir, 
                                                       label_folder=label_folder + '/' + target) 
                                                       for subgraph_file in os.listdir(datadir + '/' + target) if subgraph_file.find('gdtscore') > 0)
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
    parser.add_argument('--auto_sim_threshold', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--auto_sim_threshold_global', default=False, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    args.gpus = 1

    for sampled_data in os.listdir(args.datadir):
        
        sampled_datadir = args.datadir + '/' + sampled_data

        outdir = args.outdir + '/' + sampled_data

        all_targets = sorted(os.listdir(sampled_datadir))
        
        savedir = outdir + '/processed_data'

        dgl_folder, label_folder = generate_dgl_and_labels(savedir=savedir, targets=all_targets, datadir=sampled_datadir, 
                                                           scoredir=args.scoredir, sim_threshold=args.sim_threshold,
                                                           auto_sim_threshold=args.auto_sim_threshold,
                                                           auto_sim_threshold_global=args.auto_sim_threshold_global)

        random.shuffle(all_targets)  

        data_array = np.array(all_targets)

        folds = np.array_split(data_array, 10)

        for test_fold_index in range(len(folds)):
            
            folddir = f"{outdir}/fold{test_fold_index}"

            os.makedirs(folddir, exist_ok=True)

            print(f"Fold {test_fold_index}:")

            targets_test_in_fold = list(folds[test_fold_index])

            val_fold_index = test_fold_index + 1
            if val_fold_index >= len(folds):
                val_fold_index = 0
            
            targets_val_in_fold = list(folds[val_fold_index])

            targets_train_in_fold = []
            for i in range(len(folds)):
                if i != val_fold_index and i != test_fold_index:
                    targets_train_in_fold += list(folds[i])
            
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
