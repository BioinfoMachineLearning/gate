import os, sys, argparse, time
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists

def generate_af_scores(indir: str, 
                       outdir: str,
                       model_csv: str):

    target_dict = {'model': [], 'plddt': [], 'percentile': []}

    model_size_ratio = {}
    if model_csv is not None and os.path.exists(model_csv):    
        model_info_df = pd.read_csv(model_csv)
        model_size_ratio = dict(zip(list(model_info_df['model']), list(model_info_df['model_size_norm'])))
        target_dict['plddt_norm'] = []
        target_dict['percentile_norm'] = []

    modeldir = os.path.join(outdir , 'models')
    makedir_if_not_exists(modeldir)

    for pdb in os.listdir(indir):
        trgpdb = os.path.join(modeldir, f"{pdb}.pdb")
        os.system(f"cp {os.path.join(indir, pdb)} {trgpdb}")
        # plddt
        ppdb = PandasPdb()
        ppdb.read_pdb(trgpdb)
        plddt = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']['b_factor']
        plddt = plddt.to_numpy().astype(np.float32) / 100
        target_dict['model'] += [pdb]
        target_dict['plddt'] += [np.mean(plddt)]

        if 'plddt_norm' in target_dict:
            target_dict['plddt_norm'] += [np.mean(plddt) * float(model_size_ratio[pdb])]
    
    plddt_array = np.array(target_dict['plddt'])
    for plddt in target_dict['plddt']:
        percentile = (plddt_array < plddt).sum() / len(target_dict['model'])
        target_dict['percentile'] += [percentile]

    if 'plddt_norm' in target_dict: 
        plddt_norm_array = np.array(target_dict['plddt_norm'])
        for plddt in target_dict['plddt_norm']:
            percentile_norm = (plddt_norm_array < plddt).sum() / len(target_dict['model'])
            target_dict['percentile_norm'] += [percentile_norm]

    pd.DataFrame(target_dict).to_csv(os.path.join(outdir, 'plddt.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--model_size_csv', type=str, required=False)

    args = parser.parse_args()

    generate_af_scores(indir=args.indir, 
                       outdir=args.outdir,
                       model_csv=args.model_size_csv)

