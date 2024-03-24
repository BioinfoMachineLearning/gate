
import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import * 

def read_tmscore(infile):
    tmscore, gdtscore = 0.0, 0.0
    for line in open(infile):
        line = line.rstrip('\n')
        if len(line) > 0:
            if line.split()[0] == 'TM-score':
                tmscore = float(line.split()[2])
            if line.split()[0] == 'GDT-score':
                gdtscore = float(line.split()[2])
    return tmscore, gdtscore

def run_command(inparams):
    tmscore_program, indir, pdb1, pdb2, outdir = inparams
    cmd = f"{tmscore_program} {indir}/{pdb1} {indir}/{pdb2} > {outdir}/{pdb1}_{pdb2}.tmscore"
    # print(cmd)
    os.system(cmd)
    
def run_pairwise(tmscore_program, indir, scoredir, outfile1, outfile2, process_num):

    pdbs = sorted(os.listdir(indir))

    process_list = []
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            resultfile = os.path.join(scoredir, f"{pdb1}_{pdb2}.tmscore")
            if os.path.exists(resultfile) and len(open(resultfile).readlines()) > 15:
                continue
            process_list.append([tmscore_program, indir, pdb1, pdb2, scoredir])

    pool = Pool(processes=process_num)
    results = pool.map(run_command, process_list)
    pool.close()
    pool.join()

    tmscores_dict, gdtscores_dict = {}, {}
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            tmscore_file = os.path.join(scoredir, f"{pdb1}_{pdb2}.tmscore")
            if not os.path.exists(tmscore_file):
                raise Exception(f"cannot find {tmscore_file}")
            
            tmscore, gdtscore = read_tmscore(tmscore_file)
            tmscores_dict[f"{pdb1}_{pdb2}"] = tmscore
            gdtscores_dict[f"{pdb1}_{pdb2}"] = gdtscore

    data_dict1, data_dict2 = {}, {}
    for i in range(len(pdbs)):
        pdb1 = pdbs[i]
        tmscores, gdtscores = [], []
        for j in range(len(pdbs)):
            pdb2 = pdbs[j]
            tmscore, gdtscore = 1.0, 1.0
            if pdb1 != pdb2:
                tmscore, gdtscore = tmscores_dict[f"{pdb1}_{pdb2}"], gdtscores_dict[f"{pdb1}_{pdb2}"]

            tmscores += [tmscore]
            gdtscores += [gdtscore]

        data_dict1[pdb1] = tmscores
        data_dict2[pdb1] = gdtscores

    pd.DataFrame(data_dict1).to_csv(outfile1)
    pd.DataFrame(data_dict2).to_csv(outfile2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--tmscore_program', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--process_num', type=int, default=40)

    args = parser.parse_args()

    outfile1 = os.path.join(args.outdir, 'pairwise_tmscore.csv')
    outfile2 = os.path.join(args.outdir, 'pairwise_gdtscore.csv')

    scoredir = os.path.join(args.outdir, 'scores')
    os.makedirs(scoredir, exist_ok=True)

    run_pairwise(args.tmscore_program, args.indir, scoredir, outfile1, outfile2, args.process_num)
