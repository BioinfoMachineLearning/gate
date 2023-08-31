import argparse, os
import sys
import logging
import numpy as np
from scipy.stats import skew
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Download pdb structure from pdb bank for monomer or dimer by list"
    parser.add_argument("-i", "--indir", help="pdb name in lower case", type=str, required=True)

    args = parser.parse_args()

    for result_file in sorted(os.listdir(args.indir)):
        targetname = result_file.rstrip('.csv')
        df = pd.read_csv(args.indir + '/' + result_file, index_col=[0])
        mean_std = np.mean(np.array(df['std']))
        mean_std_norm = np.mean(np.array(df['std_norm']))
        # print(df)
        print(f"{str(mean_std)} {str(mean_std_norm)}")



