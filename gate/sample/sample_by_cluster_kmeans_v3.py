import os, sys, argparse, time
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists
from sklearn.cluster import KMeans
import random, copy
from sklearn.metrics import silhouette_score

def sample_models_by_kmeans(indir, sample_number_per_target, outdir_base):

    for pairwise_file in os.listdir(indir):
        
        targetname = pairwise_file.replace('.csv', '')

        outdir = f"{outdir_base}/{targetname}"

        makedir_if_not_exists(outdir)
        
        pairwise_graph = pd.read_csv(indir + '/' + pairwise_file, index_col=[0])

        mean_pairwise_score = np.mean(np.array(pairwise_graph))

        if mean_pairwise_score < 0.8:

            pairwise_graph_np = np.array(pairwise_graph).T

            silhouette_scores = []
            for k in range(2, 10):
                kmeans =  KMeans(n_clusters=k, random_state=0, n_init="auto").fit(pairwise_graph_np)
                cluster_labels = kmeans.fit_predict(pairwise_graph_np)
                silhouette_scores.append(silhouette_score(pairwise_graph_np, cluster_labels))

            kmeans_cluster_num = 2 + np.argmax(silhouette_scores)
            sample_number_in_cluster = int(50 / kmeans_cluster_num)

            kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0, n_init="auto").fit(pairwise_graph)

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

                selected_columns = [pairwise_graph.columns[i] for i in subgraph_indices]
                
                subgraph_df = pairwise_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_{i}.csv")
        
        else:
            
            pairwise_graph_np = np.array(pairwise_graph).T
            sample_number_in_cluster = 50
            for i in range(sample_number_per_target):
                subgraph_indices = []
                if len(pairwise_graph.columns) >= sample_number_in_cluster:
                    subgraph_indices = random.sample(range(len(pairwise_graph.columns)), k=sample_number_in_cluster) 
                else:
                    subgraph_indices = range(len(pairwise_graph.columns))

                # subgraph_indices = sorted(subgraph_indices)

                selected_columns = [pairwise_graph.columns[i] for i in subgraph_indices]
                
                subgraph_df = pairwise_graph[selected_columns].loc[subgraph_indices]

                subgraph_df.to_csv(f"{outdir}/subgraph_{i}.csv")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
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
        sample_models_by_kmeans(args.indir, sample_number_per_target, outdir)

