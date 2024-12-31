import os, sys, argparse, time
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists
from sklearn.cluster import KMeans
import random, copy
from sklearn.metrics import silhouette_score

def sample_models_by_kmeans(pairwise_usalign_file, 
                            pairwise_mmalign_file,
                            pairwise_qsscore_file,
                            pairwise_dockq_wave_file,
                            pairwise_dockq_ave_file,
                            pairwise_cad_score_file,
                            sample_number_per_target, 
                            outdir):

    # Generate a random seed
    random_seed = random.randint(0, 2**32 - 1)

    # Use this seed for further random operations
    random.seed(random_seed)

    makedir_if_not_exists(outdir)
    
    pairwise_usalign_graph = pd.read_csv(pairwise_usalign_file, index_col=[0])
    pairwise_mmalign_graph = pd.read_csv(pairwise_mmalign_file, index_col=[0])
    pairwise_qsscore_graph = pd.read_csv(pairwise_qsscore_file, index_col=[0])
    pairwise_dockq_wave_graph = pd.read_csv(pairwise_dockq_wave_file, index_col=[0])
    pairwise_dockq_ave_graph = pd.read_csv(pairwise_dockq_ave_file, index_col=[0])
    pairwise_cad_score_graph = pd.read_csv(pairwise_cad_score_file, index_col=[0])

    mean_pairwise_score = np.mean(np.array(pairwise_usalign_graph))

    model_to_cluster = None

    if mean_pairwise_score < 0.8:
        
        pairwise_usalign_graph_np = np.array(pairwise_usalign_graph).T
        pairwise_mmalign_graph_np = np.array(pairwise_mmalign_graph).T
        pairwise_qsscore_graph_np = np.array(pairwise_qsscore_graph).T
        pairwise_dockq_wave_graph_np = np.array(pairwise_dockq_wave_graph).T
        pairwise_dockq_ave_graph_np = np.array(pairwise_dockq_ave_graph).T
        pairwise_cad_score_graph_np = np.array(pairwise_cad_score_graph).T

        silhouette_scores = []
        for k in range(2, 10):
            kmeans =  KMeans(n_clusters=k, random_state=0, n_init="auto").fit(pairwise_usalign_graph_np)
            cluster_labels = kmeans.fit_predict(pairwise_usalign_graph_np)
            silhouette_scores.append(silhouette_score(pairwise_usalign_graph_np, cluster_labels))

        kmeans_cluster_num = 2 + np.argmax(silhouette_scores)
        sample_number_in_cluster = int(50 / kmeans_cluster_num)

        kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0, n_init="auto").fit(pairwise_usalign_graph_np)
        
        model_to_cluster = {pairwise_usalign_graph.columns[i]: kmeans.labels_[i] for i in range(len(pairwise_usalign_graph.columns))}

        for i in range(sample_number_per_target):
            subgraph_indices = []
            for j in range(kmeans_cluster_num):
                cluster_indices = list(np.where(kmeans.labels_ == j)[0])
                if len(cluster_indices) >= sample_number_in_cluster:
                    sampled_indices = random.sample(cluster_indices, k=sample_number_in_cluster) 
                else:
                    sampled_indices = cluster_indices

                subgraph_indices += sampled_indices

            # subgraph_indices = sorted(subgraph_indices)
            
            subgraph_indices = list(subgraph_indices)

            random.shuffle(subgraph_indices)

            selected_columns = [pairwise_usalign_graph.columns[i] for i in subgraph_indices]
            
            subgraph_df = pairwise_usalign_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_usalign_{i}.csv")

            subgraph_df = pairwise_mmalign_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_mmalign_{i}.csv")

            subgraph_df = pairwise_qsscore_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_qsscore_{i}.csv")

            subgraph_df = pairwise_dockq_wave_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_dockq_wave_{i}.csv")

            subgraph_df = pairwise_dockq_ave_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_dockq_ave_{i}.csv")

            subgraph_df = pairwise_cad_score_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_cad_score_{i}.csv")
    
    else:
        sample_number_in_cluster = 50
        for i in range(sample_number_per_target):
            subgraph_indices = []
            if len(pairwise_usalign_graph.columns) >= sample_number_in_cluster:
                subgraph_indices = random.sample(range(len(pairwise_usalign_graph.columns)), k=sample_number_in_cluster) 
            else:
                subgraph_indices = range(len(pairwise_usalign_graph.columns))

            # subgraph_indices = sorted(subgraph_indices)

            subgraph_indices = list(subgraph_indices)

            random.shuffle(subgraph_indices)

            selected_columns = [pairwise_usalign_graph.columns[i] for i in subgraph_indices]
            
            subgraph_df = pairwise_usalign_graph[selected_columns].loc[subgraph_indices]
            
            subgraph_df.to_csv(f"{outdir}/subgraph_usalign_{i}.csv")

            subgraph_df = pairwise_mmalign_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_mmalign_{i}.csv")

            subgraph_df = pairwise_qsscore_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_qsscore_{i}.csv")

            subgraph_df = pairwise_dockq_wave_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_dockq_wave_{i}.csv")

            subgraph_df = pairwise_dockq_ave_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_dockq_ave_{i}.csv")

            subgraph_df = pairwise_cad_score_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_cad_score_{i}.csv")

        model_to_cluster = {pairwise_usalign_graph.columns[i]: "0" for i in range(len(pairwise_usalign_graph.columns))}

    return model_to_cluster