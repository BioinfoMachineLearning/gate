import os, sys, argparse, time
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists

def generate_af_scores(indir: str, 
                       outdir: str, 
                       targetname: str, 
                       model_csv: str):

    model_info_df = pd.read_csv(model_csv)
    model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))

    target_dict = {'model': [], 'plddt': [], 'percentile': [], 'plddt_norm': [], 'percentile_norm': []}

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
        target_dict['plddt_norm'] += [np.mean(plddt) * float(model_size_ratio[pdb])]
    
    plddt_array = np.array(target_dict['plddt'])
    for plddt in target_dict['plddt']:
        percentile = (plddt_array < plddt).sum() / len(target_dict['model'])
        target_dict['percentile'] += [percentile]

    plddt_norm_array = np.array(target_dict['plddt_norm'])
    for plddt in target_dict['plddt_norm']:
        percentile_norm = (plddt_norm_array < plddt).sum() / len(target_dict['model'])
        target_dict['percentile_norm'] += [percentile_norm]

    pd.DataFrame(target_dict).to_csv(outdir + '/' + targetname + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--interface_dir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        makedir_if_not_exists(outdir)
        generate_af_scores(indir=args.indir + '/' + target, 
                           outdir=outdir, 
                           targetname=target, 
                           model_csv=args.interface_dir + '/' + target + '/' + target + '.csv')

