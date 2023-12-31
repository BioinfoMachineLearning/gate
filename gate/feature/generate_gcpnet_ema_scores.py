import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
from gate.tool.alignment import parse_fasta

GCPNET_EMA_program_dir = '/home/jl4mc/gate/tools/GCPNet-EMA'
GCPNET_EMA_program = '/home/jl4mc/gate/tools/GCPNet-EMA/src/predict.py'

def generate_gcpnet_scores(indir: str, 
                           outdir: str, 
                           fasta_path: str, 
                           targetname: str, 
                           model_csv: str):
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    for pdb in sorted(os.listdir(indir)):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")

    base_cmd = f"python {GCPNET_EMA_program} " \
                f"model=gcpnet_ema data=ema data.predict_batch_size=1 data.num_workers=16 " \
                f"data.python_exec_path=/home/jl4mc/mambaforge/envs/gcpnet/bin/python " \
                f"data.lddt_exec_path=/home/jl4mc/mambaforge/envs/gcpnet/bin/lddt " \
                f"data.pdbtools_dir=/home/jl4mc/mambaforge/envs/gcpnet/lib/python3.10/site-packages/pdbtools/ " \
                f"logger=csv trainer.accelerator=gpu trainer.devices=[0] " \
                f"data.predict_input_dir={modeldir} " \
                f"data.ablate_ankh_embeddings=true model.ablate_gtn=true "
    
    config_list = {'no_esm': {
                                'ckpt_path': GCPNET_EMA_program_dir + '/ckpts/structure_ema_finetuned_gcpnet_without_esm_emb_x8tjgsf4_best_epoch_027.ckpt',
                                'ablate_esm_embeddings': 'true',
                                'ablate_af2_plddt': 'false'
                                },
                    'no_plddt': {
                                'ckpt_path': GCPNET_EMA_program_dir + '/ckpts/structure_ema_finetuned_gcpnet_without_plddt_ije6iplr_best_epoch_055.ckpt',
                                'ablate_esm_embeddings': 'false',
                                'ablate_af2_plddt': 'true'
                                },
                    'esm_plddt': {
                                'ckpt_path': GCPNET_EMA_program_dir + '/ckpts/structure_ema_finetuned_gcpnet_i2d5t9xh_best_epoch_106.ckpt',
                                'ablate_esm_embeddings': 'false',
                                'ablate_af2_plddt': 'false'
                                },
                    }
    
    for config_name in config_list:

        resultfile = f"{outdir}/{config_name}.csv"

        if not os.path.exists(resultfile):
            
            os.system("rm /tmp/predicted_*.pdb")
            
            config_out_dir = outdir + '/' + config_name
            os.makedirs(config_out_dir, exist_ok=True)
            
            config = config_list[config_name]
            ckpt_path = config['ckpt_path']
            ablate_esm_embeddings = config['ablate_esm_embeddings']
            ablate_af2_plddt = config['ablate_af2_plddt']

            cmd = base_cmd + f"ckpt_path={ckpt_path} " \
                             f"data.ablate_esm_embeddings={ablate_esm_embeddings} " \
                             f"model.ablate_af2_plddt={ablate_af2_plddt} " \
                             f"data.predict_output_dir={config_out_dir}"
            try:
                os.system(cmd)
            except Exception as e:
                print(e)
                return

            if not os.path.exists(config_out_dir + '/result.csv'):
                raise Exception(f"Cannot find {config_out_dir}/result.csv!")

            os.system(f"cp {config_out_dir}/result.csv {resultfile}")

        if not os.path.exists(model_csv):
            continue
        
        if os.path.exists(f"{outdir}/{targetname}_{config_name}.csv"):
            continue
            
        model_info_df = pd.read_csv(model_csv)
        model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))

        pred_model_out_dir = outdir + '/' + config_name + '_pred_pdbs'
        os.makedirs(pred_model_out_dir, exist_ok=True)

        df = pd.read_csv(resultfile)
        models = [] # list(df['MODEL'])
        scores = [] # list(df['PRED_DOCKQ'])
        scores_norm = []
        for model, pred_model, global_score in zip(list(df['input_annotated_pdb_filepath']), list(df['predicted_annotated_pdb_filepath']), list(df['global_score'])):
            modelname = os.path.basename(model)
            modelname = modelname.replace('.pdb', '')
            if modelname not in model_size_ratio:
                continue
            models += [modelname]
            scores += [global_score / 100]
            scores_norm += [global_score / 100 * float(model_size_ratio[modelname])]
            os.system(f"cp {pred_model} {pred_model_out_dir}/{modelname}")

        for pdb in sorted(os.listdir(indir)):
            pdbname = pdb.replace('.pdb', '')
            if pdbname not in models:
                models += [pdbname]
                scores += [0.0]
                scores_norm += [0.0]
        
        pd.DataFrame({'model': models, 'score': scores, 'score_norm': scores_norm}).to_csv(f"{outdir}/{targetname}_{config_name}.csv")

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--interface_dir', type=str, required=True)
    parser.add_argument('--fastadir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)

        # if os.path.exists(outdir + '/' + target + '.csv'):
        #     continue

        generate_gcpnet_scores(indir=args.indir + '/' + target,
                                fasta_path=args.fastadir + '/' + target + '.fasta',  
                                outdir=outdir, 
                                targetname=target, 
                                model_csv=args.interface_dir + '/' + target  + '.csv')



