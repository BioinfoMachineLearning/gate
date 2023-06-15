import os, sys, argparse, time
import numpy as np
import pandas as pd
from util import makedir_if_not_exists
from sklearn.cluster import KMeans
import random, copy

def sample_models_by_kmeans(indir, kmeans_cluster_num, sample_number_in_cluster, sample_number_per_target, outdir_base):

    for pairwise_file in os.listdir(indir):
        
        targetname = pairwise_file.replace('.csv', '')

        outdir = f"{outdir_base}/{targetname}"

        makedir_if_not_exists(outdir)
        
        pairwise_graph = pd.read_csv(indir + '/' + pairwise_file, index_col=[0])

        kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0, n_init="auto").fit(np.array(pairwise_graph))

        for i in range(sample_number_per_target):
            subgraph_indices = []
            for j in range(kmeans_cluster_num):
                cluster_indices = list(np.where(kmeans.labels_ == j)[0])
                if len(cluster_indices) >= sample_number_in_cluster:
                    sampled_indices = random.sample(cluster_indices, k=sample_number_in_cluster) 
                else:
                    sampled_indices = cluster_indices

                subgraph_indices += sampled_indices

            subgraph_indices = sorted(subgraph_indices)

            selected_columns = [pairwise_graph.columns[i] for i in subgraph_indices]
            
            subgraph_df = pairwise_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_{i}.csv")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    # parser.add_argument('--kmeans_cluster_num', type=int, required=True)
    # parser.add_argument('--sample_number_in_cluster', type=int, required=True)
    # parser.add_argument('--sample_number_per_target', type=int, required=True)

    seed = 111
    random.seed(seed)

    args = parser.parse_args()

    kmeans_cluster_nums = range(5, 11)

    sample_numbers_in_cluster = {'5': [10, 15, 20],
                                '6': [10, 15, 20],
                                '7': [10, 15],
                                '8': [10, 15],
                                '9': [10],
                                '10': [10]}
                                
    sample_numbers_per_target = [100, 200]

    for cluster_num in kmeans_cluster_nums:
        sample_numbers = sample_numbers_in_cluster[str(cluster_num)]
        for sample_number in sample_numbers:
            for sample_number_per_target in sample_numbers_per_target:
                outdir = f"{args.outdir}/k{cluster_num}_n{sample_number}_t{sample_number_per_target}"
                makedir_if_not_exists(outdir)
                sample_models_by_kmeans(args.indir, cluster_num, sample_number, sample_number_per_target, outdir)

