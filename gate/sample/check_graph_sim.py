import os, sys, argparse, time
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists
from sklearn.cluster import KMeans
import random, copy
from sklearn.metrics import silhouette_score

def check_sim(indir):

    sampled_graph_nodes = []

    sim_num = 0

    for infile in os.listdir(indir):

        if infile.find("usalign") < 0:
            continue

        subgraph_usalign_df = pd.read_csv(indir + '/' + infile, index_col=[0])

        models = sorted(list(subgraph_usalign_df.columns))

        model_str = ','.join(models)
        
        if model_str not in sampled_graph_nodes:
            sampled_graph_nodes += [model_str]
        else:
            sim_num += 1
    
    return sim_num


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    args = parser.parse_args()

    for target in os.listdir(args.indir):
        print(f"checking {target}")
        sim_num = check_sim(args.indir + '/' + target)
        if sim_num > 0:
            print(f"{target}\t{sim_num}")
