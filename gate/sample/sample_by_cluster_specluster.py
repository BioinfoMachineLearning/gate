import os, sys, argparse, time
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists
from sklearn.cluster import SpectralClustering
import random, copy

def find_asymmetry(matrix):
    transpose = matrix.T
    asymmetry_indices = np.where(matrix != transpose)
    return asymmetry_indices

def sample_models_by_SpectralClustering(indir, cluster_num, sample_number_in_cluster, sample_number_per_target, outdir_base):
    
    for pairwise_file in os.listdir(indir):
        
        targetname = pairwise_file.replace('_sym.csv', '')

        outdir = f"{outdir_base}/{targetname}"

        makedir_if_not_exists(outdir)
        
        pairwise_graph = pd.read_csv(indir + '/' + pairwise_file, index_col=[0])

        pairwise_graph_array = 1 - np.array(pairwise_graph)

        pairwise_graph_array = np.exp(- pairwise_graph_array ** 2 / (2. * 1.0 ** 2))

        spec_cluster = SpectralClustering(n_clusters=cluster_num, random_state=0, affinity='precomputed').fit(pairwise_graph_array)

        sampled_models_indices = set()

        for i in range(sample_number_per_target):
            subgraph_indices = []
            for j in range(cluster_num):
                cluster_indices = list(np.where(spec_cluster.labels_ == j)[0])
                if len(cluster_indices) >= sample_number_in_cluster:
                    sampled_indices = random.sample(cluster_indices, k=sample_number_in_cluster) 
                else:
                    sampled_indices = cluster_indices

                subgraph_indices += sampled_indices

            subgraph_indices = sorted(subgraph_indices)

            selected_columns = [pairwise_graph.columns[i] for i in subgraph_indices]
            
            subgraph_df = pairwise_graph[selected_columns].loc[subgraph_indices]

            subgraph_df.to_csv(f"{outdir}/subgraph_{i}.csv")

            sampled_models_indices.update(subgraph_indices)

        if len(sampled_models_indices) != len(pairwise_graph.columns):
            print(f"Some models are not sampled for {targetname}")
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    seed = 111

    random.seed(seed)

    args = parser.parse_args()

    spec_cluster_cluster_nums = range(5, 11)

    sample_numbers_in_cluster = {'5': [10]}
                                #'6': [10, 15],
                                #'7': [10, 15],
                                #'8': [10],
                                #'9': [10],
                                #{'10': [10]}
                                
    sample_numbers_per_target = [1000]

    for cluster_num in spec_cluster_cluster_nums:
        sample_numbers = sample_numbers_in_cluster[str(cluster_num)]
        for sample_number in sample_numbers:
            for sample_number_per_target in sample_numbers_per_target:
                outdir = f"{args.outdir}/k{cluster_num}_n{sample_number}_t{sample_number_per_target}"
                makedir_if_not_exists(outdir)
                print(f"sampling {outdir}")
                sample_models_by_SpectralClustering(args.indir, cluster_num, sample_number, sample_number_per_target, outdir)

