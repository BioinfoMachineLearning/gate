import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
import itertools
from gate.tool.hhblits import HHBlits
from gate.tool.jackhmmer import Jackhmmer
from gate.tool.protein import get_sequence, split_pdb
from gate.tool.alignment import *
from gate.tool.species_interact_v3 import Species_interact_v3

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

hhblits_databases = ['/home/bml_casp15/BML_CASP15/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt']
jackhmmer_database = '/home/bml_casp15/BML_CASP15/databases/uniref90/uniref90.fasta' 
hhblits_binary_path = '/home/bml_casp15/anaconda3/envs/bml_casp15/bin/hhblits'
jackhmmer_binary = '/home/bml_casp15/anaconda3/envs/bml_casp15/bin/jackhmmer'

cdpred_program = '/home/jl4mc/gate/gate/feature/run_cdpred.sh'

def run_msa_tool(inparams):
    msa_runner, input_fasta_path, msa_out_path = inparams
    msa_runner.query(input_fasta_path, msa_out_path)

class Chain:
    def __init__(self, sequence, count):
        self.sequence = sequence
        self.count = count

def run_cdpred_on_dimers(inparams):

    name1, name2, chain_pdbs, a3m, outdir = inparams

    if len(chain_pdbs) == 1:
        cmd = f"sh {cdpred_program} {name1}_{name2} {chain_pdbs[0]} {a3m} homodimer {outdir}"
    else:
        cmd = f"sh {cdpred_program} {name1}_{name2} '{' '.join(chain_pdbs)}' {a3m} heterodimer {outdir}"

    try:
        print(cmd)
        os.system(cmd)
    except Exception as e:
        print(e)


def generate_cmap_cdpred(targetname, fasta_path, outdir, pairwise_score_csv, model_dir):

    sequences, descriptions = parse_fasta(open(fasta_path).read())

    chain_counter = 0
    sequence_id_map = {}
    for sequence in sequences:
        if sequence not in sequence_id_map:
            sequence_id_map[sequence] = dict(chain_id=PDB_CHAIN_IDS[chain_counter], count=1)
            chain_counter += 1
        else:
            sequence_id_map[sequence]['count'] += 1

    stoichiometry = ''.join([sequence_id_map[sequence]['chain_id'] + str(sequence_id_map[sequence]['count']) 
                            for sequence in sequence_id_map])
    print("Detected stoichiometry: " + stoichiometry)
    
    # Extract dimers
    print("Start to generate monomer alignments...")
    unique_sequences = list(sequence_id_map.keys())
    msa_process_list = []
    for sequence in unique_sequences:
        workdir = f"{outdir}/{sequence_id_map[sequence]['chain_id']}"
        makedir_if_not_exists(workdir)

        monomer_fasta = workdir + '/' + sequence_id_map[sequence]['chain_id'] + '.fasta'
        with open(monomer_fasta, 'w') as fw:
            fw.write(f">{sequence_id_map[sequence]['chain_id']}\n")
            fw.write(f"{sequence}\n")

        a3mfile = monomer_fasta.replace('.fasta', '.a3m')
        if not os.path.exists(a3mfile) or len(open(a3mfile).readlines()) == 0:
            hhblits_runner = HHBlits(binary_path=hhblits_binary_path, databases=hhblits_databases)
            msa_process_list.append([hhblits_runner, monomer_fasta, a3mfile])
           
        stofile = monomer_fasta.replace('.fasta', '.sto')
        if not os.path.exists(stofile) or len(open(stofile).readlines()) == 0:
            jackhmmer_runner = Jackhmmer(binary_path=jackhmmer_binary, database_path=jackhmmer_database)
            msa_process_list.append([jackhmmer_runner, monomer_fasta, stofile])


    pool = Pool(processes=15)
    results = pool.map(run_msa_tool, msa_process_list)
    pool.close()
    pool.join()

    # find the model with the highest pairwise score - average similarity scores by column
    print("Searching the suitable model based on pairwise scores...")
    reference_model_dir = outdir + '/refer_model'
    makedir_if_not_exists(reference_model_dir)

    pairwise_df = pd.read_csv(pairwise_score_csv, index_col=[0])
    models = pairwise_df.columns
    tmscores = np.array([np.mean(np.array(pairwise_df[model])) for model in models])
    chain_pdbs = {}
    while True:
        select_model_idx = np.argmax(tmscores)
        selected_model = models[select_model_idx]
        print(f"Checking {selected_model}")
        chain_models = split_pdb(model_dir + '/' + selected_model, reference_model_dir)
        # need to pair the chain pdb and sequence
        for sequence in unique_sequences:
            for chain_model in chain_models:
                pdb_sequence = get_sequence(chain_model)
                if pdb_sequence == sequence:
                    chain_pdbs[sequence_id_map[sequence]['chain_id']] = chain_model
                    break

        if len(chain_pdbs) == len(unique_sequences):
            print(f"Sucess!")
            break
        else:
            chain_pdbs = {}
            tmscores[select_model_idx] = 0.0
            print(f"Cannot map the sequences: {len(chain_pdbs)} and {len(unique_sequences)}!")
    
    run_cdpred_list = []
    # homodimers
    print("Processing homodimers...")
    for sequence in unique_sequences:
        if int(sequence_id_map[sequence]['count']) >= 2:
            chain_id = sequence_id_map[sequence]['chain_id']
            workdir = f"{outdir}/{chain_id}2/"
            makedir_if_not_exists(workdir)
            run_cdpred_list.append([targetname + chain_id, targetname + chain_id, [chain_pdbs[chain_id]], f"{outdir}/{chain_id}/{chain_id}.a3m", workdir])
            #run_cdpred_on_dimers([targetname + chain_id, targetname + chain_id, [chain_pdbs[chain_id]], f"{outdir}/{chain_id}/{chain_id}.a3m", workdir])

    # heterodimers
    print("Processing heterodimers...")
    for i in range(len(unique_sequences)):
        sequence1 = unique_sequences[i]
        for j in range(i+1, len(unique_sequences)):
            sequence2 = unique_sequences[j]
            
            chain_id1 = sequence_id_map[sequence1]['chain_id']
            chain_id2 = sequence_id_map[sequence2]['chain_id']

            workdir = f"{outdir}/{chain_id1}_{chain_id2}"
            makedir_if_not_exists(workdir)

            paired_a3m = pair_a3m(f"{outdir}/{chain_id1}/{chain_id1}.sto", f"{outdir}/{chain_id2}/{chain_id2}.sto", workdir)

            run_cdpred_list.append([targetname + chain_id1, targetname + chain_id2, [chain_pdbs[chain_id1], chain_pdbs[chain_id2]], paired_a3m, workdir])
            # run_cdpred_on_dimers([targetname + chain_id1, targetname + chain_id2, [chain_pdbs[chain_id1], chain_pdbs[chain_id2]], paired_a3m, workdir])

    pool = Pool(processes=5)
    results = pool.map(run_cdpred_on_dimers, run_cdpred_list)
    pool.close()
    pool.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fastadir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--pairwise_dir', type=str, required=True)

    args = parser.parse_args()

    for fastafile in sorted(os.listdir(args.fastadir)):
        targetname = fastafile.replace('.fasta', '')
        print(f"Processing {targetname}")

        pairwise_score_csv = args.pairwise_dir + '/' + targetname + '.csv'
        if not os.path.exists(pairwise_score_csv):
            print(f"Cannot find the pairwise score file: {pairwise_score_csv}")
            continue

        outdir = args.outdir + '/' + targetname
        makedir_if_not_exists(outdir)

        generate_cmap_cdpred(targetname, args.fastadir + '/' + fastafile, outdir, pairwise_score_csv, args.model_dir + '/' + targetname)

