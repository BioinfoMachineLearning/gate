import argparse, os
import sys
import logging
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Download pdb structure from pdb bank for monomer or dimer by list"
    parser.add_argument("-i1", "--indir1", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-i2", "--indir2", help="pdb name in lower case", type=str, required=True)

    args = parser.parse_args()

    for result_file in sorted(os.listdir(args.indir2)):
        #print(result_file)
        df1 = pd.read_csv(args.indir1 + '/' + result_file, index_col=[0])
        df2 = pd.read_csv(args.indir2 + '/' + result_file, index_col=[0])
        if result_file[0] == "T":
            if list(df1['model'])[0].find('o') < 0:
                df1['model'] = df1['model'] + 'o'
            if list(df2['model'])[0].find('o') < 0:
                df2['model'] = df2['model'] + 'o'
        #print(df2)
        df = df1.merge(df2, on=f'model', how="inner")
        #print(df)
        
        tmscore1 = np.array(df['tmscore_x'])
        tmscore2 = np.array(df['tmscore_y'])
        corr = pearsonr(tmscore1, tmscore2)[0]
        print(f"{result_file.replace('.csv', '')} {str(corr)}")



