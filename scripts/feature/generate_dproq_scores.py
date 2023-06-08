import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from util import *
import re, subprocess

dproqa_dir = '/home/bml_casp15/tools/DProQA/'
dproqa_program = 'inference.py'

def generate_dproqa_scores(indir, outdir):

    os.chdir(dproqa_dir)
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    pdbs = sorted(os.listdir(indir))
    for pdb in pdbs:
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    cmd = f"python {dproqa_program} -c {modeldir} --w {outdir}/work -r {outdir}"
    try:
        print(cmd)
        # os.system(cmd)
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

        generate_dproqa_scores(args.indir + '/' + target, outdir)

