import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from util import *
import re, subprocess

EnQA_dir = '/home/bml_casp15/tools/EnQA/'
EnQA_program = 'EnQA-MSA.py'

def mergePDB(inputPDB, outputPDB, newStart=1):
    with open(inputPDB, 'r') as f:
        x = f.readlines()
    filtered = [i for i in x if re.match(r'^ATOM.+', i)]
    chains = set([i[21] for i in x if re.match(r'^ATOM.+', i)])
    chains = list(chains)
    chains.sort()
    with open(outputPDB + '.tmp', 'w') as f:
        f.writelines(filtered)
    merge_cmd = 'pdb_selchain -{} {} | pdb_chain -A | pdb_reres -{} > {}'.format(','.join(chains),
                                                                                 outputPDB + '.tmp',
                                                                                 newStart,
                                                                                 outputPDB)
    subprocess.run(args=merge_cmd, shell=True)
    os.remove(outputPDB + '.tmp')


def generate_enqa_scores(indir, outdir):

    os.chdir(EnQA_dir)
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    pdbs = sorted(os.listdir(indir))
    for pdb in pdbs:
        if os.path.exists(outdir + '/' + pdb + '.npy'):
            continue
        mergePDB(indir + '/' + pdb, modeldir + '/' + pdb + '.pdb')
        cmd = f"python {EnQA_program} --input {modeldir}/{pdb}.pdb --output {outdir}"
        try:
            print(cmd)
            os.system(cmd)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)

        generate_enqa_scores(args.indir + '/' + target, outdir)

