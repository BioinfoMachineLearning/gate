
import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from util import *

def read_mmalign(infile):
    for line in open(infile):
        line = line.rstrip('\n')
        if len(line) > 0:
            if line.split()[0] == 'TM-score=' and line.find('Structure_2') > 0:
                tmscore = float(line.split()[1])
                return tmscore
    return 0

def run_command(inparams):
    mmalign_program, indir, pdb1, pdb2, outdir = inparams
    cmd = f"{mmalign_program} {indir}/{pdb1} {indir}/{pdb2} > {outdir}/{pdb1}_{pdb2}.mmalign"
    print(cmd)
    os.system(cmd)
    
def run_pairwise(mmalign_program, indir, scoredir, outfile):

    pdbs = sorted(os.listdir(indir))

    process_list = []
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            if os.path.exists(f"{scoredir}/{pdb1}_{pdb2}.mmalign"):
                continue
            process_list.append([mmalign_program, indir, pdb1, pdb2, scoredir])

    pool = Pool(processes=80)
    results = pool.map(run_command, process_list)
    pool.close()
    pool.join()

    scores_dict = {}
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            mmalign_file1 = f"{scoredir}/{pdb1}_{pdb2}.mmalign"
            if not os.path.exists(mmalign_file1):
                raise Exception(f"cannot find {mmalign_file1}")
            tmscore1 = read_mmalign(mmalign_file1)

            mmalign_file2 = f"{scoredir}/{pdb2}_{pdb1}.mmalign"
            if not os.path.exists(mmalign_file2):
                raise Exception(f"cannot find {mmalign_file2}")
            tmscore2 = read_mmalign(mmalign_file2)

            scores_dict[f"{pdb1}_{pdb2}"] = min(tmscore1, tmscore2)

    data_dict = {}
    for i in range(len(pdbs)):
        pdb1 = pdbs[i]
        tmscores = []
        for j in range(len(pdbs)):
            pdb2 = pdbs[j]
            tmscore = 1
            if pdb1 != pdb2:
                tmscore = scores_dict[f"{pdb1}_{pdb2}"]
            tmscores += [tmscore]
        data_dict[pdb1] = tmscores

    pd.DataFrame(data_dict).to_csv(outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--mmalign_program', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        if os.path.exists(args.outdir + '/' + target + '_sym.csv'):
            continue

        scoredir = args.outdir + '/' + target
        makedir_if_not_exists(scoredir)

        outfile = args.outdir + '/' + target + '_sym.csv'

        run_pairwise(args.mmalign_program, args.indir + '/' + target, scoredir, outfile)
