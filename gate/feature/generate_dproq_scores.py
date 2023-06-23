import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
from subprocess import call

dproqa_dir = '/home/bml_casp15/tools/DProQA/'
dproqa_program = 'inference.py'

def generate_dproqa_scores(indir, outdir, targetname, model_csv):

    model_info_df = pd.read_csv(model_csv)
    model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))

    os.chdir(dproqa_dir)
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    for pdb in sorted(os.listdir(indir)):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    resultfile = outdir + '/Ranking.csv'
    if not os.path.exists(resultfile):
        cmd = f"python {dproqa_program} -c {modeldir} --w {outdir}/work -r {outdir}"
        try:
            call([cmd], shell=True)
        except Exception as e:
            print(e)
            return

        if not os.path.exists(resultfile):
            raise Exception(f"Cannot find {resultfile}!")

    df = pd.read_csv(resultfile)
    models = list(df['MODEL'])
    scores = list(df['PRED_DOCKQ'])
    scores_norm = []
    for model, dproqa_score in zip(models, scores):
        scores_norm += [dproqa_score * float(model_size_ratio[model])]

    for pdb in sorted(os.listdir(indir)):
        pdbname = pdb.replace('.pdb', '')
        if pdbname not in models:
            models += [pdbname]
            scores += [0.0]
            scores_norm += [0.0]
    
    pd.DataFrame({'model': models, 'DockQ': scores, 'DockQ_norm': scores_norm}).to_csv(outdir + '/' + targetname + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--interface_dir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)

        generate_dproqa_scores(args.indir + '/' + target, outdir, target, interface_dir + '/' + target + '.csv')

