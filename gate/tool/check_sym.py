import os, sys, argparse, time
import difflib
import sys
import pandas as pd
import numpy as np

def is_symmetric(arr):
    # Check if the array is symmetric using numpy.allclose
    return np.allclose(arr, arr.T)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    args = parser.parse_args()
    
    for infile in os.listdir(args.indir):
        print("checking " + infile)
        df = pd.read_csv(args.indir + '/' + infile, index_col=[0])
        if is_symmetric(np.array(df)):
            print('yes')

