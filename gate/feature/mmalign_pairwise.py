
import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import * 

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
    # print(cmd)
    os.system(cmd)
    
def run_pairwise(mmalign_program, indir, scoredir, outfile, process_num):

    pdbs = sorted(os.listdir(indir))

    process_list = []
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            resultfile = f"{scoredir}/{pdb1}_{pdb2}.mmalign"
            if os.path.exists(resultfile) and len(open(resultfile).readlines()) > 15:
                continue
            process_list.append([mmalign_program, indir, pdb1, pdb2, scoredir])

    pool = Pool(processes=process_num)
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
            mmalign_file = f"{scoredir}/{pdb1}_{pdb2}.mmalign"
            if not os.path.exists(mmalign_file):
                raise Exception(f"cannot find {mmalign_file}")
            scores_dict[f"{pdb1}_{pdb2}"] = read_mmalign(mmalign_file)

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
    parser.add_argument('--process_num', type=int, default=120)

    args = parser.parse_args()

    scoredir = os.path.join(args.outdir, 'scores')
    os.makedirs(scoredir, exist_ok=True)

    run_pairwise(mmalign_program=args.mmalign_program, 
                 indir=args.indir, 
                 scoredir=scoredir, 
                 outfile=os.path.join(args.outdir, 'pairwise_mmalign.csv'),
                 process_num=args.process_num)
