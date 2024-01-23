
import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import * 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):

        resultfile = args.outdir + '/' + target + '.csv'
        full_graph_df = pd.read_csv(resultfile, index_col=[0])
        models = full_graph_df.columns

        avg_tmscores = []
        for model in models:
            avg_tmscores += [np.mean(np.array(full_graph_df[model]))]

        outfile = args.outdir + '/' + target + '_avg.csv'
        pd.DataFrame({'model': models, 'score': avg_tmscores}).to_csv(outfile)
