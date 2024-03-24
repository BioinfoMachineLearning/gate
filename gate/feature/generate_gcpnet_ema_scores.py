import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
from gate.tool.alignment import parse_fasta

GCPNET_EMA_program = 'src/predict.py'

def generate_gcpnet_scores(gcpnet_ema_env_path: str,
                           gcpnet_ema_program_path: str,
                           indir: str, 
                           outdir: str, 
                           model_csv: str):
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    for pdb in sorted(os.listdir(indir)):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    os.chdir(gcpnet_ema_program_path)
    cmd = f"{gcpnet_ema_env_path}/bin/python {GCPNET_EMA_program} " \
                f"model=gcpnet_ema data=ema data.predict_batch_size=1 data.num_workers=16 " \
                f"data.python_exec_path=/home/jl4mc/mambaforge/envs/gcpnet/bin/python " \
                f"data.lddt_exec_path=/home/jl4mc/mambaforge/envs/gcpnet/bin/lddt " \
                f"data.pdbtools_dir=/home/jl4mc/mambaforge/envs/gcpnet/lib/python3.10/site-packages/pdbtools/ " \
                f"logger=csv trainer.accelerator=gpu trainer.devices=[0] " \
                f"data.predict_input_dir={modeldir} " \
                f"data.ablate_ankh_embeddings=true model.ablate_gtn=true " \
                f"ckpt_path=checkpoints/structure_ema_finetuned_gcpnet_i2d5t9xh_best_epoch_106.ckpt " \
                f"data.ablate_esm_embeddings=false " \
                f"model.ablate_af2_plddt=false " \
                f"data.predict_output_dir={outdir}/workdir"
    
    resultfile = f"{outdir}/result.csv"

    if not os.path.exists(resultfile):
        pattern = r"(/[^/\s]+)+\.csv"
        logfile = os.path.join(outdir, 'run.log')
        os.system(f"{cmd} &> {logfile}")
        for line in open(logfile):
            if line.find("Predictions saved to:") >= 0:
                result_csv = line.split('Predictions saved to:')[1]
                match = re.search(pattern, result_csv)
                os.system(f"cp {match.group(0)} {resultfile}")
        

    data_dict = {'model': [], 'score': []}
    model_size_ratio = {}
    if model_csv is not None and os.path.exists(model_csv):                
        model_info_df = pd.read_csv(model_csv)
        model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))
        data_dict['score_norm'] = []

    pred_model_out_dir = os.path.join(outdir, 'pred_pdbs')
    os.makedirs(pred_model_out_dir, exist_ok=True)

    df = pd.read_csv(resultfile)
    for model, pred_model, global_score in zip(list(df['input_annotated_pdb_filepath']), list(df['predicted_annotated_pdb_filepath']), list(df['global_score'])):
        modelname = os.path.basename(model)
        modelname = modelname.replace('.pdb', '')
        data_dict['model'] += [modelname]
        data_dict['score'] += [global_score / 100]
        
        if 'score_norm' in data_dict:
            data_dict['score_norm'] += [global_score / 100 * float(model_size_ratio[modelname])]

        os.system(f"cp {pred_model} {pred_model_out_dir}/{modelname}")

    for pdb in sorted(os.listdir(indir)):
        pdbname = pdb.replace('.pdb', '')
        if pdbname not in data_dict['model']:
            data_dict['model'] += [pdbname]
            data_dict['score'] += [0.0]
            if 'score_norm' in data_dict:
                data_dict['score_norm'] += [0.0]
    
    pd.DataFrame(data_dict).to_csv(os.path.join(outdir, 'esm_plddt.csv'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--model_size_csv', type=str, required=False)
    parser.add_argument('--gcpnet_ema_program_path', type=str, required=True)
    parser.add_argument('--gcpnet_ema_env_path', type=str, required=True)

    args = parser.parse_args()

    generate_gcpnet_scores(gcpnet_ema_env_path=args.gcpnet_ema_env_path,
                           gcpnet_ema_program_path=args.gcpnet_ema_program_path,
                           indir=args.indir,
                           outdir=args.outdir, 
                           model_csv=args.model_size_csv)



