import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
import itertools
from gate.tool.protein import *
from gate.tool.alignment import parse_fasta

def generate_interface_model_size(targetname: str, 
                                  fasta_path: str, 
                                  input_model_dir: str, 
                                  cdpreddir: str, 
                                  outdir: str):

    # read sequences from fasta file
    sequences, descriptions = parse_fasta(open(fasta_path).read())

    target_length = np.sum(np.array([len(sequence) for sequence in sequences]))

    data_dict = {'model': [], 'interface_size_norm': [], 'model_size_norm': []}
    for model in sorted(os.listdir(input_model_dir)):

        chain_pdb_dir = cdpreddir + '/models/' + model + '/monomer_pdbs' 

        cmap_files = read_files_by_prefix_and_ext(indir=chain_pdb_dir, ext='cmap')
        # print(cmap_files)
        interface_size = 0
        for cmap_file in cmap_files:
            model_cmap = np.loadtxt(cmap_file)
            interface_size += 2 * (model_cmap > 0).sum()

        aligned_model_size = 0
        aligned_pdb_files = read_files_by_prefix_and_ext(indir=chain_pdb_dir, ext='aligned')
        print(aligned_pdb_files)
        for aligned_pdb_file in aligned_pdb_files:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('', aligned_pdb_file)
            chain_id = list(structure[0].child_dict.keys())
            xyzPDB = structure[0][chain_id[0]]
            aligned_model_size += len(xyzPDB)

        data_dict['model'] += [model]
        data_dict['interface_size_norm'] += [interface_size / target_length]
        data_dict['model_size_norm'] += [aligned_model_size / target_length]

    pd.DataFrame(data_dict).to_csv(outdir + '/' + targetname + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fastadir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--cdpreddir', type=str, required=True)

    args = parser.parse_args()

    for fastafile in sorted(os.listdir(args.fastadir)):
        
        targetname = fastafile.replace('.fasta', '')
        print(f"Processing {targetname}")

        outdir = args.outdir + '/' + targetname
        makedir_if_not_exists(outdir)

        if os.path.exists(outdir + '/' + targetname + '.csv'):
            continue
            
        generate_interface_model_size(targetname=targetname, 
                                      fasta_path=args.fastadir + '/' + fastafile, 
                                      input_model_dir=args.modeldir + '/' + targetname,
                                      cdpreddir=args.cdpreddir + '/' + targetname,
                                      outdir=outdir)

