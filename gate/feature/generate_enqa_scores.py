import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
from gate.tool.alignment import parse_fasta

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

def generate_enqa_scores(indir: str, 
                         outdir: str, 
                         fasta_path: str, 
                         targetname: str, 
                         model_csv: str):

    model_info_df = pd.read_csv(model_csv)
    model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))

    max_length_threshold = 2500
    # read sequences from fasta file
    sequences, descriptions = parse_fasta(open(fasta_path).read())
    target_length = np.sum(np.array([len(sequence) for sequence in sequences]))
    target_dict = {'model': [], 'score': [], 'score_norm': []}
    if target_length >= max_length_threshold:
        for model in model_size_ratio:
            target_dict['model'] += [model]
            target_dict['score'] += [0.0]
            target_dict['score_norm'] += [0.0]

    else:
        os.chdir(EnQA_dir)
        
        modeldir = outdir + '/models'
        makedir_if_not_exists(modeldir)

        for pdb in sorted(os.listdir(indir)):
            resultfile = outdir + '/' + pdb + '.npy'
            if not os.path.exists(resultfile):
                mergePDB(indir + '/' + pdb, modeldir + '/' + pdb + '.pdb')
                cmd = f"python {EnQA_program} --input {modeldir}/{pdb}.pdb --output {outdir}/{pdb}"
                try:
                    print(cmd)
                    os.system(cmd)
                    os.system(f"cp {outdir}/{pdb}/{pdb}.npy {outdir}/{pdb}.npy")
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
                target_dict['score_norm'] += [global_score * float(model_size_ratio[pdb])]
            else:
                target_dict['score'] += [0.0]
                target_dict['score_norm'] += [0.0]
        
    pd.DataFrame(target_dict).to_csv(outdir + '/' + targetname + '.csv')
    

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

        generate_enqa_scores(indir=args.indir + '/' + target,
                             fasta_path=args.fastadir + '/' + target + '.fasta',  
                             outdir=outdir, 
                             targetname=target, 
                             model_csv=args.interface_dir + '/' + target + '/' + target + '.csv')



