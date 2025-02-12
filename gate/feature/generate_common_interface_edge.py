import os, sys, argparse, time
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd
from gate.tool.utils import makedir_if_not_exists
from gate.tool.protein import read_files_by_prefix_and_ext

def generate_common_interface_edge(indir: str, 
                                   outdir: str, 
                                   cdpreddir: str):

    pdbs = sorted(os.listdir(indir))

    data_dict = {}
    for i in range(len(pdbs)):
        
        number_of_common_interfaces = []

        pdb1 = pdbs[i]

        chain_pdb_dir1 = os.path.join(cdpreddir, 'models', pdb1, 'monomer_pdbs')

        _cmap_files1 = read_files_by_prefix_and_ext(indir=chain_pdb_dir1, ext='cmap', full_path=False)

        _interfaces1 = set([cmap_file1[0:2] for cmap_file1 in _cmap_files1])

        for j in range(len(pdbs)):

            pdb2 = pdbs[j]

            if pdb1 == pdb2:

                number_of_common_interfaces += [len(_interfaces1)]

            else:

                chain_pdb_dir2 = os.path.join(cdpreddir, 'models', pdb2, 'monomer_pdbs')

                _cmap_files2 = read_files_by_prefix_and_ext(indir=chain_pdb_dir2, ext='cmap', full_path=False)

                _interfaces2 = set([cmap_file2[0:2] for cmap_file2 in _cmap_files2])

                number_of_common_interfaces += [len([1 for _interface2 in _interfaces2 if _interface2 in _interfaces1])]

        data_dict[pdb1] = number_of_common_interfaces

    pd.DataFrame(data_dict).to_csv(os.path.join(outdir, 'common_interface.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--cdpreddir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    generate_common_interface_edge(indir=args.indir,
                                   outdir=args.outdir, 
                                   cdpreddir=args.cdpreddir)

