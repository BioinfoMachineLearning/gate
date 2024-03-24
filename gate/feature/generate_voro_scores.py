import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess

voroqa_program = 'ftdmp-qa-all'

def generate_voro_scores(voro_env_path: str,
                         voro_program_path:str,
                         indir: str, 
                         outdir: str,
                         model_csv: str):

    os.chdir(voro_program_path)
    
    modeldir = os.path.join(outdir, 'models')
    makedir_if_not_exists(modeldir)

    for pdb in sorted(os.listdir(indir)):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    resultfile = outdir + '/result.csv'
    if not os.path.exists(resultfile):

        current_path = os.environ.get('PATH', '')
        new_path = os.pathsep.join([os.path.join(voro_env_path, 'bin'), current_path])
        # Set the updated PATH variable
        os.environ['PATH'] = new_path

        cmd = f"ls {modeldir}/*.pdb | ./{voroqa_program} --conda-path {voro_env_path} " \
            f"--workdir '{outdir}/tmp' --rank-names protein_protein_voromqa_and_global_and_gnn_no_sr > {resultfile}"
        try:
            print(cmd)
            os.system(cmd)
        except Exception as e:
            print(e)
    
        if not os.path.exists(resultfile):
            raise Exception(f"Cannot find {resultfile}!")
    print(model_csv)
    if not os.path.exists(model_csv):
        return

    model_info_df = pd.read_csv(model_csv)
    model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))

    df = pd.read_csv(resultfile, sep=' ')
    # df['GNN_sum_score'] = (df['GNN_sum_score'] - df['GNN_sum_score'].min()) / (df['GNN_sum_score'].max() - df['GNN_sum_score'].min()) 

    models = [] # list(df['ID'])
    gnn_scores = [] # list(df['GNN_sum_score'])
    gnn_pcad_scores = [] # list(df['GNN_pcadscore'])
    dark_scores = [] # list(df['voromqa_dark'])

    gnn_scores_norm = []
    gnn_pcad_scores_norm = []
    dark_scores_norm = []

    for model, gnn_score, gnn_pcad_score, dark_score in zip(list(df['ID']), list(df['GNN_sum_score']), list(df['GNN_pcadscore']), list(df['voromqa_dark'])):
        model = model.replace('.pdb', '')
        if model not in model_size_ratio:
            continue
        models += [model]
        gnn_scores += [gnn_score]
        gnn_pcad_scores += [gnn_pcad_score]
        dark_scores += [dark_score]
        gnn_scores_norm += [gnn_score * float(model_size_ratio[model])]
        gnn_pcad_scores_norm += [gnn_pcad_score * float(model_size_ratio[model])]
        dark_scores_norm += [dark_score * float(model_size_ratio[model])]

    for pdb in sorted(os.listdir(indir)):
        if pdb not in models:
            models += [pdb]
            gnn_scores += [0.0]
            gnn_pcad_scores += [0.0]
            dark_scores += [0.0]

            gnn_scores_norm += [0.0]
            gnn_pcad_scores_norm += [0.0]
            dark_scores_norm += [0.0]
    
    pd.DataFrame({'model': models, 'GNN_sum_score': gnn_scores,  'GNN_sum_score_norm': gnn_scores_norm,
                  'GNN_pcadscore': gnn_pcad_scores, 'GNN_pcadscore_norm': gnn_pcad_scores_norm,
                  'voromqa_dark': dark_scores, 'voromqa_dark_norm': dark_scores_norm}).to_csv(os.path.join(outdir, 'voro.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--model_size_csv', type=str, required=True)
    parser.add_argument('--voro_program_path', type=str, required=True) 
    parser.add_argument('--voro_env_path', type=str, required=True) 

    args = parser.parse_args()

    generate_voro_scores(voro_env_path=args.voro_env_path,
                         voro_program_path=args.voro_program_path,
                         indir=args.indir, 
                         outdir=args.outdir,
                         model_csv=args.model_size_csv)

