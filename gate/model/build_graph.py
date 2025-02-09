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
import random

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


def build_multimer_model_graph(subgraph_file: str,
                                features_multimer,
                                out_dgl_file: str,
                                sim_threshold: float,
                                use_af_feature: bool,
                                use_interface_pairwise: bool,
                                use_gcpnet_ema: bool) -> None:
    
    if not os.path.exists(subgraph_file):
        raise FileNotFoundError(f'Cannot not find subgraph: {subgraph_file} ')

    # print(f'Processing {filename}')
    scaler = MinMaxScaler()

    subgraph_usalign_df = pd.read_csv(subgraph_file, index_col=[0])
    subgraph_mmalign_df = pd.read_csv(subgraph_file.replace('usalign', 'mmalign'), index_col=[0])
    subgraph_qsscore_df = pd.read_csv(subgraph_file.replace('usalign', 'qsscore'), index_col=[0])
    subgraph_dockq_wave_df = pd.read_csv(subgraph_file.replace('usalign', 'dockq_wave'), index_col=[0])
    subgraph_dockq_ave_df = pd.read_csv(subgraph_file.replace('usalign', 'dockq_ave'), index_col=[0])
    subgraph_cad_score_df = pd.read_csv(subgraph_file.replace('usalign', 'cad_score'), index_col=[0])

    nodes_num = len(subgraph_usalign_df)

    models = subgraph_usalign_df.columns

    # node features
    # 1. alphafold global plddt score
    alphafold_scores_df = pd.read_csv(features_multimer.plddt)
    alphafold_scores_dict = {k: v for k, v in zip(list(alphafold_scores_df['model']), list(alphafold_scores_df['plddt_norm']))}
    alphafold_plddt_scores = [alphafold_scores_dict[model] for model in models]
    alphafold_plddt_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_plddt_scores).reshape(-1, 1))).float()
    
    alphafold_confidence_score_feature, alphafold_num_inter_pae_feature, alphafold_iptm_feature, alphafold_dockq_feature = None, None, None, None
    if use_af_feature:
        # b. alphafold confidence score, iptm score, mpDockQ score
        alphafold_scores_df = pd.read_csv(features_multimer.af_features)
        alphafold_confidence_score_dict = {k: v for k, v in zip(list(alphafold_scores_df['jobs']), list(alphafold_scores_df['iptm_ptm']))}
        alphafold_confidence_scores = [alphafold_confidence_score_dict[model] for model in models]
        alphafold_confidence_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_confidence_scores).reshape(-1, 1))).float()

        alphafold_num_inter_pae_dict = {k: v for k, v in zip(list(alphafold_scores_df['jobs']), list(alphafold_scores_df['num_inter_pae']))}
        alphafold_num_inter_paes = [alphafold_num_inter_pae_dict[model] for model in models]
        alphafold_num_inter_pae_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_num_inter_paes).reshape(-1, 1))).float()

        alphafold_iptm_dict = {k: v for k, v in zip(list(alphafold_scores_df['jobs']), list(alphafold_scores_df['iptm']))}
        alphafold_iptms = [alphafold_iptm_dict[model] for model in models]
        alphafold_iptm_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_iptms).reshape(-1, 1))).float()

        alphafold_dockq_dict = {k: v for k, v in zip(list(alphafold_scores_df['jobs']), list(alphafold_scores_df['mpDockQ/pDockQ']))}
        alphafold_dockqs = [alphafold_dockq_dict[model] for model in models]
        alphafold_dockq_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_dockqs).reshape(-1, 1))).float()

    # pairwise_usalign
    average_sim_scores_in_subgraph = [np.mean(np.array(subgraph_usalign_df[model])) for model in models]
    average_sim_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_subgraph).reshape(-1, 1))).float()

    full_usalign_graph_df = pd.read_csv(features_multimer.pairwise_usalign, index_col=[0])
    average_sim_scores_in_full_graph = [np.mean(np.array(full_usalign_graph_df[model])) for model in models]
    average_sim_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_full_graph).reshape(-1, 1))).float()

    # pairwise_mmalign
    average_sim_mmalign_scores_in_subgraph = [np.mean(np.array(subgraph_mmalign_df[model])) for model in models]
    average_sim_mmalign_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_mmalign_scores_in_subgraph).reshape(-1, 1))).float()

    full_mmalign_graph_df = pd.read_csv(features_multimer.pairwise_mmalign, index_col=[0])
    average_sim_mmalign_scores_in_full_graph = [np.mean(np.array(full_mmalign_graph_df[model])) for model in models]
    average_sim_mmalign_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_mmalign_scores_in_full_graph).reshape(-1, 1))).float()

    # pairwise_qsscore
    average_sim_qsscores_in_subgraph = [np.mean(np.array(subgraph_qsscore_df[model])) for model in models]
    average_sim_qsscore_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_qsscores_in_subgraph).reshape(-1, 1))).float()

    full_qsscore_graph_df = pd.read_csv(features_multimer.pairwise_qsscore, index_col=[0])
    average_sim_qsscores_in_full_graph = [np.mean(np.array(full_qsscore_graph_df[model])) for model in models]
    average_sim_qsscore_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_qsscores_in_full_graph).reshape(-1, 1))).float()

    # pairwise_dockq_wave
    average_sim_dockq_waves_in_subgraph = [np.mean(np.array(subgraph_dockq_wave_df[model])) for model in models]
    average_sim_dockq_wave_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_dockq_waves_in_subgraph).reshape(-1, 1))).float()

    full_dockq_wave_graph_df = pd.read_csv(features_multimer.pairwise_dockq_wave, index_col=[0])
    average_sim_dockq_waves_in_full_graph = [np.mean(np.array(full_dockq_wave_graph_df[model])) for model in models]
    average_sim_dockq_wave_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_dockq_waves_in_full_graph).reshape(-1, 1))).float()

    # pairwise_dockq_ave
    average_sim_dockq_aves_in_subgraph = [np.mean(np.array(subgraph_dockq_ave_df[model])) for model in models]
    average_sim_dockq_ave_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_dockq_aves_in_subgraph).reshape(-1, 1))).float()

    full_dockq_ave_graph_df = pd.read_csv(features_multimer.pairwise_dockq_ave, index_col=[0])
    average_sim_dockq_aves_in_full_graph = [np.mean(np.array(full_dockq_ave_graph_df[model])) for model in models]
    average_sim_dockq_ave_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_dockq_aves_in_full_graph).reshape(-1, 1))).float()

    # pairwise_cad_score
    average_sim_cad_scores_in_subgraph = [np.mean(np.array(subgraph_cad_score_df[model])) for model in models]
    average_sim_cad_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_cad_scores_in_subgraph).reshape(-1, 1))).float()

    full_cad_score_graph_df = pd.read_csv(features_multimer.pairwise_cad_score, index_col=[0])
    average_sim_cad_scores_in_full_graph = [np.mean(np.array(full_cad_score_graph_df[model])) for model in models]
    average_sim_cad_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_cad_scores_in_full_graph).reshape(-1, 1))).float()

    # voro scores: gnn, gnn_pcadscore, voromqa_dark
    voro_scores_df = pd.read_csv(features_multimer.voro)
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
    dproqa_scores_df = pd.read_csv(features_multimer.dproqa)
    dproqa_scores_dict = {k: v for k, v in zip(list(dproqa_scores_df['model']), list(dproqa_scores_df['DockQ_norm']))}

    dproqa_scores = [dproqa_scores_dict[model] for model in models]
    dproqa_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(dproqa_scores).reshape(-1, 1))).float()

    # contact scores by cdpred
    contact_scores_df = pd.read_csv(features_multimer.icps)
    icps_scores_dict = {k: v for k, v in zip(list(contact_scores_df['model']), list(contact_scores_df['icps']))}
    recall_scores_dict = {k: v for k, v in zip(list(contact_scores_df['model']), list(contact_scores_df['recall']))}

    icps_scores = [icps_scores_dict[model] for model in models]
    recall_scores = [recall_scores_dict[model] for model in models]

    icps_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(icps_scores).reshape(-1, 1))).float()
    recall_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(recall_scores).reshape(-1, 1))).float()

    # enqa scores
    enqa_scores_df = pd.read_csv(features_multimer.enqa)
    enqa_scores_dict = {k: v for k, v in zip(list(enqa_scores_df['model']), list(enqa_scores_df['score_norm']))}

    enqa_scores = [enqa_scores_dict[model] for model in models]

    # Target too large to run enqa
    if np.sum(np.array(enqa_scores)) == 0:
        enqa_score_feature = torch.tensor(enqa_scores).reshape(-1, 1).float()
    else:
        enqa_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(enqa_scores).reshape(-1, 1))).float()

    # gcpnet scores
    gcpnet_scores_df = pd.read_csv(features_multimer.gcpnet_ema)
    gcpnet_scores_dict = {k: v for k, v in zip(list(gcpnet_scores_df['model']), list(gcpnet_scores_df['score_norm']))}

    gcpnet_scores = [gcpnet_scores_dict[model] for model in models]
    gcpnet_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(gcpnet_scores).reshape(-1, 1))).float()

    # edge features
    # a. global fold similarity between two models
    # b. number of common interfaces
    subgraph_usalign_array = np.array(subgraph_usalign_df)
    subgraph_mmalign_array = np.array(subgraph_mmalign_df)
    subgraph_qsscore_array = np.array(subgraph_qsscore_df)
    subgraph_dockq_wave_array = np.array(subgraph_dockq_wave_df)
    subgraph_dockq_ave_array = np.array(subgraph_dockq_ave_df)
    subgraph_cad_score_array = np.array(subgraph_cad_score_df)

    common_interface_array = np.array(pd.read_csv(features_multimer.common_interface, index_col=[0]))
    
    # print(sim_threshold)
    src_nodes, dst_nodes = [], []
    edge_sim, edge_mmalign_sim, edge_qsscore_sim, edge_common_interface = [], [], [], []
    edge_dockq_wave_sim, edge_dockq_ave_sim, edge_cad_score_sim = [], [], []
    for src in range(nodes_num):
        for dst in range(nodes_num):
            if src == dst:
                continue
            if subgraph_mmalign_array[src, dst] >= sim_threshold:
                src_nodes += [src]
                dst_nodes += [dst]

                # edge_sim += [subgraph_mmalign_array[dst, src]] # non-symmetric matrix, the similarity score should be noramlized by the target model
                edge_sim += [subgraph_usalign_array[src, dst]] # should be normalized by the source model? e.g., source model is larger
                edge_mmalign_sim += [subgraph_mmalign_array[src, dst]]
                edge_qsscore_sim += [subgraph_qsscore_array[src, dst]]
                edge_dockq_wave_sim += [subgraph_dockq_wave_array[src, dst]]
                edge_dockq_ave_sim += [subgraph_dockq_ave_array[src, dst]]
                edge_cad_score_sim += [subgraph_cad_score_array[src, dst]]
                edge_common_interface += [common_interface_array[src, dst]]


    node_features = [alphafold_plddt_score_feature]
        
    if use_af_feature:

        node_features.append(alphafold_confidence_score_feature)
        node_features.append(alphafold_num_inter_pae_feature)
        node_features.append(alphafold_iptm_feature)
        node_features.append(alphafold_dockq_feature)

    node_features.append(average_sim_score_in_subgraph_feature)
    node_features.append(average_sim_score_in_full_graph_feature)
    node_features.append(average_sim_mmalign_score_in_subgraph_feature)
    node_features.append(average_sim_mmalign_score_in_full_graph_feature)
    node_features.append(average_sim_qsscore_in_subgraph_feature)
    node_features.append(average_sim_qsscore_in_full_graph_feature)

    if use_interface_pairwise:
        node_features.append(average_sim_dockq_wave_in_subgraph_feature)
        node_features.append(average_sim_dockq_wave_in_full_graph_feature)
        node_features.append(average_sim_dockq_ave_in_subgraph_feature)
        node_features.append(average_sim_dockq_ave_in_full_graph_feature)
        node_features.append(average_sim_cad_score_in_subgraph_feature)
        node_features.append(average_sim_cad_score_in_full_graph_feature)
    
    node_features.append(voro_gnn_score_feature)
    node_features.append(voro_gnn_pcadscore_feature)
    node_features.append(voro_dark_score_feature)
    node_features.append(dproqa_score_feature)
    node_features.append(icps_score_feature)
    node_features.append(recall_score_feature)
    node_features.append(enqa_score_feature)

    if use_gcpnet_ema:
        node_features.append(gcpnet_score_feature)

    if len(edge_sim) > 0:
        # 6. add feature to graph
        graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), num_nodes=nodes_num)
        
        edge_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_sim).reshape(-1, 1))).float()
        edge_mmalign_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_mmalign_sim).reshape(-1, 1))).float()
        edge_qsscore_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_qsscore_sim).reshape(-1, 1))).float()
        edge_dockq_wave_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_dockq_wave_sim).reshape(-1, 1))).float()
        edge_dockq_ave_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_dockq_ave_sim).reshape(-1, 1))).float()
        edge_cad_score_sim_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_cad_score_sim).reshape(-1, 1))).float()
        edge_common_interface_feature = torch.tensor(scaler.fit_transform(torch.tensor(edge_common_interface).reshape(-1, 1))).float()

        edge_features = [edge_sim_feature, edge_mmalign_sim_feature, edge_qsscore_sim_feature]
        if use_interface_pairwise:
            edge_features.append(edge_dockq_wave_sim_feature)
            edge_features.append(edge_dockq_ave_sim_feature)
            edge_features.append(edge_cad_score_sim_feature)

        edge_features.append(edge_common_interface_feature)
        
        # edge_sin_pos = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)
        update_node_feature(graph, node_features)
        update_edge_feature(graph, edge_features)

    else:
        graph = dgl.DGLGraph()
        graph.add_nodes(nodes_num)
        update_node_feature(graph, node_features)

    dgl.save_graphs(filename=out_dgl_file, g_list=graph)
    # print(f'{filename}\nSUCCESS')
    return None


def multimer_graph_wrapper(subgraph_file: str, features_multimer, out_dgl_file, sim_threshold, use_af_feature, use_interface_pairwise, use_gcpnet_ema):
    build_multimer_model_graph(subgraph_file=subgraph_file,
                               features_multimer=features_multimer,
                               out_dgl_file=out_dgl_file,
                               sim_threshold=sim_threshold,
                               use_af_feature=use_af_feature,
                               use_interface_pairwise=use_interface_pairwise,
                               use_gcpnet_ema=use_gcpnet_ema)


def generate_multimer_dgls(sample_dir,
                           dgl_dir,
                           features_multimer,
                           sim_threshold,
                           use_af_feature,
                           use_interface_pairwise,
                           use_gcpnet_ema):

    os.makedirs(dgl_dir, exist_ok=True)
    Parallel(n_jobs=-1)(delayed(multimer_graph_wrapper)(subgraph_file=os.path.join(sample_dir, subgraph_file),
                                                        features_multimer=features_multimer,
                                                        out_dgl_file=os.path.join(dgl_dir, subgraph_file.replace('.csv', '.dgl')),
                                                        sim_threshold=sim_threshold,
                                                        use_af_feature=use_af_feature,
                                                        use_interface_pairwise=use_interface_pairwise,
                                                        use_gcpnet_ema=use_gcpnet_ema)
                                                        for subgraph_file in os.listdir(sample_dir) if subgraph_file.find('usalign') > 0)


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


def build_monomer_model_graph(targetname, subgraph_file: str, features_monomer, out_dgl_file, sim_threshold) -> None:

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
    alphafold_scores_df = pd.read_csv(features_monomer.plddt)
    alphafold_scores_dict = {k: v for k, v in zip(list(alphafold_scores_df['model']), list(alphafold_scores_df['plddt']))}
    alphafold_plddt_scores = [alphafold_scores_dict[model] for model in models]
    
    alphafold_plddt_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(alphafold_plddt_scores).reshape(-1, 1))).float()
    
    # gdtscore
    # b1. average pairwise similarity score in graph
    # b2. average pairwise similarity score for all models

    average_sim_scores_in_subgraph = [np.mean(np.array(subgraph_gdtscore_df[model])) for model in models]
    average_sim_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_subgraph).reshape(-1, 1))).float()

    full_gdtscore_graph_df = pd.read_csv(features_monomer.pairwise_gdtscore, index_col=[0])
    average_sim_scores_in_full_graph = [np.mean(np.array(full_gdtscore_graph_df[model])) for model in models]
    average_sim_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_scores_in_full_graph).reshape(-1, 1))).float()

    # tmscore
    average_sim_tmscore_in_subgraph = [np.mean(np.array(subgraph_tmscore_df[model])) for model in models]
    average_sim_tmscore_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_tmscore_in_subgraph).reshape(-1, 1))).float()

    full_tmscore_graph_df = pd.read_csv(features_monomer.pairwise_tmscore, index_col=[0])
    average_sim_tmscore_in_full_graph = [np.mean(np.array(full_tmscore_graph_df[model])) for model in models]
    average_sim_tmscore_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_tmscore_in_full_graph).reshape(-1, 1))).float()

    # cad-score
    average_sim_cad_scores_in_subgraph = [np.mean(np.array(subgraph_cad_score_df[model])) for model in models]
    average_sim_cad_score_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_cad_scores_in_subgraph).reshape(-1, 1))).float()

    full_cad_score_graph_df = pd.read_csv(features_monomer.pairwise_cad_score, index_col=[0])
    average_sim_cad_scores_in_full_graph = [np.mean(np.array(full_cad_score_graph_df[model])) for model in models]
    average_sim_cad_score_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_cad_scores_in_full_graph).reshape(-1, 1))).float()

    # lddt
    average_sim_lddt_in_subgraph = [np.mean(np.array(subgraph_lddt_df[model])) for model in models]
    average_sim_lddt_in_subgraph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_lddt_in_subgraph).reshape(-1, 1))).float()

    full_lddt_graph_df = pd.read_csv(features_monomer.pairwise_lddt, index_col=[0])
    average_sim_lddt_in_full_graph = [np.mean(np.array(full_lddt_graph_df[model])) for model in models]
    average_sim_lddt_in_full_graph_feature = torch.tensor(scaler.fit_transform(torch.tensor(average_sim_lddt_in_full_graph).reshape(-1, 1))).float()

    # To be added: enqa scores
    enqa_scores_df = pd.read_csv(features_monomer.enqa)
    enqa_scores_dict = {k: v for k, v in zip(list(enqa_scores_df['model']), list(enqa_scores_df['score']))}

    enqa_scores = [enqa_scores_dict[model] for model in models]

    # Target too large to run enqa
    if np.sum(np.array(enqa_scores)) == 0:
        enqa_score_feature = torch.tensor(enqa_scores).reshape(-1, 1).float()
    else:
        enqa_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(enqa_scores).reshape(-1, 1))).float()

    # d. gcpnet scores
    gcpnet_scores_df = pd.read_csv(features_monomer.gcpnet_ema)
    gcpnet_scores_dict = {k: v for k, v in zip(list(gcpnet_scores_df['model']), list(gcpnet_scores_df['score']))}

    gcpnet_scores = [gcpnet_scores_dict[model] for model in models]
    gcpnet_score_feature = torch.tensor(scaler.fit_transform(torch.tensor(gcpnet_scores).reshape(-1, 1))).float()


    # DeepRank3 features

    ranking_files = []

    ranking_files += [features_monomer.deeprank3_cluster]
    ranking_files += [features_monomer.deeprank3_singleqa]
    ranking_files += [features_monomer.deeprank3_singleqa_lite]

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
        ranking_files += [os.path.join(features_monomer.deeprank3_features, deeprank3_feature_name + '.' + targetname)]

    deeprank3_features = read_deeprank3_features(targetname, ranking_files, models)

    # edge features
    # a. global fold similarity between two models
    # b. number of common interfaces
    subgraph_gdtscore_array = np.array(subgraph_gdtscore_df)
    subgraph_tmscore_array = np.array(subgraph_tmscore_df)
    subgraph_cad_score_array = np.array(subgraph_cad_score_df)
    subgraph_lddt_array = np.array(subgraph_lddt_df)

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

    dgl.save_graphs(filename=out_dgl_file, g_list=graph)
    # print(f'{filename}\nSUCCESS')
    return None



def monomer_graph_wrapper(targetname, subgraph_file: str, features_monomer, out_dgl_file, sim_threshold):
    build_monomer_model_graph(targetname=targetname,
                              subgraph_file=subgraph_file,
                              features_monomer=features_monomer,
                              out_dgl_file=out_dgl_file,
                              sim_threshold=sim_threshold)

def generate_monomer_dgls(targetname,
                          sample_dir,
                          dgl_dir,
                          features_monomer,
                          sim_threshold):

    os.makedirs(dgl_dir, exist_ok=True)
    Parallel(n_jobs=-1)(delayed(monomer_graph_wrapper)(targetname=targetname,
                                                       subgraph_file=os.path.join(sample_dir, subgraph_file),
                                                       features_monomer=features_monomer,
                                                       out_dgl_file=os.path.join(dgl_dir, subgraph_file.replace('.csv', '.dgl')),
                                                       sim_threshold=sim_threshold)
                                                       for subgraph_file in os.listdir(sample_dir) if subgraph_file.find('gdtscore') > 0)
