import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess

voro_dir = '/home/bml_casp15/tools/ftdmp/'
voroqa_program = 'ftdmp-qa-all'

def generate_voro_scores(indir, outdir, targetname):

    os.chdir(voro_dir)
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    for pdb in sorted(os.listdir(indir)):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    resultfile = outdir + '/result.csv'
    if not os.path.exists(resultfile):
        cmd = f"ls {modeldir}/*.pdb | ./{voroqa_program} --conda-path {voro_dir}/miniconda3/ " \
            f"--workdir '{outdir}/tmp' --rank-names protein_protein_voromqa_and_global_and_gnn_no_sr > {resultfile}"
        try:
            print(cmd)
            os.system(cmd)
        except Exception as e:
            print(e)
    
        if not os.path.exists(resultfile):
            raise Exception(f"Cannot find {resultfile}!")

    df = pd.read_csv(resultfile, sep=' ')
    df['GNN_sum_score'] = (df['GNN_sum_score'] - df['GNN_sum_score'].min()) / (df['GNN_sum_score'].max() - df['GNN_sum_score'].min()) 

    models = list(df['ID'])
    gnn_scores = list(df['GNN_sum_score'])
    gnn_pcad_scores = list(df['GNN_pcadscore'])
    dark_scores = list(df['voromqa_dark'])

    for pdb in sorted(os.listdir(indir)):
        if pdb + '.pdb' not in models:
            models += [pdb + '.pdb']
            gnn_scores += [0.0]
            gnn_pcad_scores += [0.0]
            dark_scores += [0.0]
    
    pd.DataFrame({'model': models, 'GNN_sum_score': gnn_scores, 'GNN_pcadscore': gnn_pcad_scores, 'voromqa_dark': dark_scores}).to_csv(outdir + '/' + targetname + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)

        generate_voro_scores(args.indir + '/' + target, outdir, target)

