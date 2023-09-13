import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd


def read_tmscore_content(infile):
    print(infile)
    tmscore = 0
    for line in open(infile):
        line = line.rstrip('\n')
        if len(line) == 0:
            continue
        contents = line.split()
        if contents[0] == "TM-score=":
            tmscore = float(contents[1])
    print(tmscore)
    return str(tmscore)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        data = {'model': [], 'tmscore': []}
        for pdb in os.listdir(args.indir + '/' + target):
            if pdb.find('_out') < 0:
                continue
            pdbname = pdb.rstrip('_filtered_out')
            tmscore = read_tmscore_content(args.indir + '/' + target + '/' + pdb)
            if tmscore == "0":
                continue
            data['model'] += [pdbname.replace('.pdb', '')]
            data['tmscore'] += [tmscore]
        pd.DataFrame(data).to_csv(args.outdir + '/' + target + '.csv')
