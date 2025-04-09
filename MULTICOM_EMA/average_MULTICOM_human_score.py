
import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--incsv', type=str, required=True)
    parser.add_argument('--outcsv', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.incsv)
    name='pairwise_cad_score.gcpnet_esm_plddt_norm.gate'
    average_methods = name.split('.')
    averaged_predictions = list(df[average_methods].mean(axis=1))
    outdict = {'model': [], 'score': []}
    for model, score in zip(df['model'], averaged_predictions):
        outdict['model'] += [model]
        outdict['score'] += [score]
    final_df = pd.DataFrame(outdict)
    final_df = final_df.sort_values(by=['score'], ascending=False)
    final_df.reset_index(inplace=True, drop=True)
    final_df.to_csv(args.outcsv)
    