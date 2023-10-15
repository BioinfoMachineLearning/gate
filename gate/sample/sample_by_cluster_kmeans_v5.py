import os, sys, argparse, time
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists
from sklearn.cluster import KMeans
import random, copy
from sklearn.metrics import silhouette_score

def sample_models_by_kmeans(usaligndir, mmaligndir, qsscoredir, sample_number_per_target, outdir_base):

    for pairwise_file in os.listdir(usaligndir):
        if pairwise_file.find('.csv') < 0:
            continue

        targetname = pairwise_file.replace('.csv', '')

        outdir = f"{outdir_base}/{targetname}"

        makedir_if_not_exists(outdir)
        
        pairwise_usalign_graph = pd.read_csv(usaligndir + '/' + pairwise_file, index_col=[0])

        pairwise_mmalign_graph = pd.read_csv(mmaligndir + '/' + pairwise_file, index_col=[0])
        
        pairwise_qsscore_graph = pd.read_csv(qsscoredir + '/' + pairwise_file, index_col=[0])

        mean_pairwise_score = np.mean(np.array(pairwise_usalign_graph))

        if mean_pairwise_score < 0.8:
            
            pairwise_usalign_graph_np = np.array(pairwise_usalign_graph).T
            pairwise_mmalign_graph_np = np.array(pairwise_mmalign_graph).T
            pairwise_qsscore_graph_np = np.array(pairwise_qsscore_graph).T

            silhouette_scores = []
            for k in range(2, 10):
                kmeans =  KMeans(n_clusters=k, random_state=0, n_init="auto").fit(pairwise_usalign_graph_np)
                cluster_labels = kmeans.fit_predict(pairwise_usalign_graph_np)
                silhouette_scores.append(silhouette_score(pairwise_usalign_graph_np, cluster_labels))

            kmeans_cluster_num = 2 + np.argmax(silhouette_scores)
            sample_number_in_cluster = int(50 / kmeans_cluster_num)

            kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0, n_init="auto").fit(pairwise_usalign_graph_np)

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

                selected_columns = [pairwise_usalign_graph.columns[i] for i in subgraph_indices]
                
                subgraph_df = pairwise_usalign_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_usalign_{i}.csv")

                subgraph_df = pairwise_mmalign_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_mmalign_{i}.csv")

                subgraph_df = pairwise_qsscore_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_qsscore_{i}.csv")
        
        else:
            sample_number_in_cluster = 50
            for i in range(sample_number_per_target):
                subgraph_indices = []
                if len(pairwise_usalign_graph.columns) >= sample_number_in_cluster:
                    subgraph_indices = random.sample(range(len(pairwise_usalign_graph.columns)), k=sample_number_in_cluster) 
                else:
                    subgraph_indices = range(len(pairwise_usalign_graph.columns))

                # subgraph_indices = sorted(subgraph_indices)

                selected_columns = [pairwise_usalign_graph.columns[i] for i in subgraph_indices]
                
                subgraph_df = pairwise_usalign_graph[selected_columns].loc[subgraph_indices]
                
                subgraph_df.to_csv(f"{outdir}/subgraph_usalign_{i}.csv")

                subgraph_df = pairwise_mmalign_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_mmalign_{i}.csv")

                subgraph_df = pairwise_qsscore_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_qsscore_{i}.csv")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--usaligndir', type=str, required=True)
    parser.add_argument('--mmaligndir', type=str, required=True)
    parser.add_argument('--qsscoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--sample_num', type=int, required=True, default=2000)
    # parser.add_argument('--kmeans_cluster_num', type=int, required=True)
    # parser.add_argument('--sample_number_in_cluster', type=int, required=True)
    # parser.add_argument('--sample_number_per_target', type=int, required=True)

    seed = 111
    random.seed(seed)

    args = parser.parse_args()

    sample_numbers_per_target = [args.sample_num]

    for sample_number_per_target in sample_numbers_per_target:
        outdir = f"{args.outdir}/kmeans_sil_t{sample_number_per_target}"
        makedir_if_not_exists(outdir)
        sample_models_by_kmeans(args.usaligndir, args.mmaligndir, args.qsscoredir, sample_number_per_target, outdir)

