import os, sys, argparse, time
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists

def generate_af_scores(indir, outdir, targetname):
    target_dict = {'model': [], 'plddt': []}

    modeldir = outdir + '/models'
    makedir_if_not_exists(modeldir)

    for pdb in os.listdir(indir):
        os.system(f"cp {indir}/{pdb} {modeldir}/{pdb}.pdb")
        # plddt
        ppdb = PandasPdb()
        ppdb.read_pdb(modeldir + '/' + pdb + '.pdb')
        plddt = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']['b_factor']
        plddt = plddt.to_numpy().astype(np.float32) / 100
        target_dict['model'] += [pdb]
        target_dict['plddt'] += [np.mean(plddt)]
    pd.DataFrame(target_dict).to_csv(outdir + '/' + targetname + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)
        generate_af_scores(args.indir + '/' + target, outdir, target)

