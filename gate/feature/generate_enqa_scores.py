import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
from gate.tool.alignment import parse_fasta

EnQA_program = 'EnQA-MSA.py'

def mergePDB(enqa_env_path, inputPDB, outputPDB, newStart=1):
    with open(inputPDB, 'r') as f:
        x = f.readlines()
    filtered = [i for i in x if re.match(r'^ATOM.+', i)]
    chains = set([i[21] for i in x if re.match(r'^ATOM.+', i)])
    chains = list(chains)
    chains.sort()
    with open(outputPDB + '.tmp', 'w') as f:
        f.writelines(filtered)
    merge_cmd = f"{enqa_env_path}/bin/pdb_selchain -{','.join(chains)} {outputPDB}.tmp | {enqa_env_path}/bin/pdb_chain -A | {enqa_env_path}/bin/pdb_reres -{newStart} > {outputPDB}"
    print(merge_cmd)
    subprocess.run(args=merge_cmd, shell=True)
    os.remove(outputPDB + '.tmp')

def generate_enqa_scores(enqa_env_path:str,
                         enqa_program_path:str,
                         indir: str, 
                         outdir: str, 
                         fasta_path: str,
                         model_csv: str,
                         mode: str):
    
    target_dict = {'model': [], 'score': []}

    model_size_ratio = {}
    if model_csv is not None and os.path.exists(model_csv):
        model_info_df = pd.read_csv(model_csv)
        model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))
        target_dict['score_norm'] = []

    max_length_threshold = 2500
    # read sequences from fasta file
    sequences, descriptions = parse_fasta(open(fasta_path).read())
    target_length = np.sum(np.array([len(sequence) for sequence in sequences]))
    if target_length >= max_length_threshold:
        for model in sorted(os.listdir(indir)):
            target_dict['model'] += [model]
            target_dict['score'] += [0.0]
            if 'score_norm' in target_dict:
                target_dict['score_norm'] += [0.0]

    else:
        os.chdir(enqa_program_path)
        
        modeldir = os.path.join(outdir, 'models')
        os.makedirs(modeldir, exist_ok=True)

        for pdb in sorted(os.listdir(indir)):
            out_npy_dir = os.path.join(outdir, 'npys')
            os.makedirs(out_npy_dir, exist_ok=True)

            resultfile = os.path.join(out_npy_dir, pdb, pdb + '.npy')
            if not os.path.exists(resultfile):
                srcpdb = os.path.join(indir, pdb)
                trgpdb = os.path.join(modeldir, pdb + '.pdb')
                if mode == "multimer":
                    mergePDB(enqa_env_path, srcpdb, trgpdb)
                else:
                    os.system(f"cp {srcpdb} {trgpdb}")
                    
                cmd = f"{enqa_env_path}/bin/python {EnQA_program} --input {trgpdb} --output {out_npy_dir}/{pdb}"
                try:
                    print(cmd)
                    os.system(cmd)
                except Exception as e:
                    print(e)

            target_dict['model'] += [pdb]
            if os.path.exists(resultfile):
                plddt_scores = np.load(resultfile)
                if np.argwhere(np.isnan(plddt_scores)).shape[0] == 0:
                    global_score = np.mean(plddt_scores)
                else:
                    global_score = 0.0
                    print(f"There are nan values in {resultfile}")

                target_dict['score'] += [global_score]

                if 'score_norm' in target_dict:
                    target_dict['score_norm'] += [global_score * float(model_size_ratio[pdb])]
            else:
                target_dict['score'] += [0.0]

                if 'score_norm' in target_dict:
                    target_dict['score_norm'] += [0.0]

    pd.DataFrame(target_dict).to_csv(os.path.join(outdir, 'enqa.csv'))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--model_size_csv', type=str, required=False)
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--enqa_program_path', type=str, required=True)
    parser.add_argument('--enqa_env_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default="multimer", required=False)

    args = parser.parse_args()

    generate_enqa_scores(enqa_env_path=args.enqa_env_path,
                         enqa_program_path=args.enqa_program_path,
                         indir=args.indir,
                         fasta_path=args.fasta_path,
                         outdir=args.outdir,
                         model_csv=args.model_size_csv,
                         mode=args.mode)



