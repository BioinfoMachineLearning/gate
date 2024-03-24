import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
from subprocess import call

dproqa_program = 'inference.py'

def generate_dproqa_scores(dproqa_env_path:str,
                           dproqa_program_path:str,
                           indir: str, 
                           outdir: str, 
                           model_csv: str):

    os.chdir(dproqa_program_path)
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    for pdb in sorted(os.listdir(indir)):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    resultfile = outdir + '/Ranking.csv'
    if not os.path.exists(resultfile):
        
        current_path = os.environ.get('PATH', '')
        new_path = os.pathsep.join([os.path.join(dproqa_env_path, 'bin'), current_path])
        # Set the updated PATH variable
        os.environ['PATH'] = new_path

        cmd = f"{dproqa_env_path}/bin/python {dproqa_program} -c {modeldir} -w {outdir}/work -r {outdir}"
        try:
            # call([cmd], shell=True)
            print(cmd)
            os.system(cmd)
        except Exception as e:
            print(e)
            return

        if not os.path.exists(resultfile):
            raise Exception(f"Cannot find {resultfile}!")

    if not os.path.exists(model_csv):
        return
        
    model_info_df = pd.read_csv(model_csv)
    model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))

    df = pd.read_csv(resultfile)
    models = [] # list(df['MODEL'])
    scores = [] # list(df['PRED_DOCKQ'])
    scores_norm = []
    for model, dproqa_score in zip(list(df['MODEL']), list(df['PRED_DOCKQ'])):
        if model not in model_size_ratio:
            continue
        models += [model]
        scores += [dproqa_score]
        scores_norm += [dproqa_score * float(model_size_ratio[model])]

    for pdb in sorted(os.listdir(indir)):
        pdbname = pdb.replace('.pdb', '')
        if pdbname not in models:
            models += [pdbname]
            scores += [0.0]
            scores_norm += [0.0]
    
    pd.DataFrame({'model': models, 'DockQ': scores, 'DockQ_norm': scores_norm}).to_csv(os.path.join(outdir, 'dproqa.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--dproqa_program_path', type=str, required=True)
    parser.add_argument('--dproqa_env_path', type=str, required=True)   
    parser.add_argument('--model_size_csv', type=str, required=True)

    args = parser.parse_args()

    generate_dproqa_scores(dproqa_env_path=args.dproqa_env_path,
                           dproqa_program_path=args.dproqa_program_path,
                           indir=args.indir,
                           outdir=args.outdir, 
                           model_csv=args.model_size_csv)

