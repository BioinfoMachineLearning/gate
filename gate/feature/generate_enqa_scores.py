import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from gate.tool.utils import *
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


def generate_enqa_scores(indir, outdir, targetname):

    os.chdir(EnQA_dir)
    
    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    target_dict = {'model': [], 'score': []}
    
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
            global_score = np.mean(plddt_scores)
            target_dict['score'] += [global_score]
        else:
            target_dict['score'] += [0.0]
    
    pd.DataFrame(target_dict).to_csv(outdir + '/' + targetname + '.csv')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)

        if not os.path.exists(outdir + '/' + target + '.csv'):
            generate_enqa_scores(args.indir + '/' + target, outdir, target)

