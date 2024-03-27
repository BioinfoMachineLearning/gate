import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
import itertools
from gate.tool.protein import *
from gate.tool.alignment import *
import copy
from scipy.optimize import linear_sum_assignment
import json

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

class Chain:
    def __init__(self, sequence, count):
        self.sequence = sequence
        self.count = count

def read_clust_aln_file(infile):
    casp_align_seq = ""
    pdb_align_seq = ""
    for line in open(infile):
        if line[0:4] == "CASP":
            casp_align_seq += line.split()[1].rstrip('\n')
        if line[0:3] == "PDB":
            pdb_align_seq += line.split()[1].rstrip('\n')

    return casp_align_seq, pdb_align_seq


def align_sequence(clustalw_program, casp_seq, pdb_seq, outfile):

    with open(outfile + '.aln', 'w') as fw: 
        fw.write(f"%CASP\n{casp_seq}\n%PDB\n{pdb_seq}")

    cmd = f"{clustalw_program}  -MATRIX=BLOSUM -TYPE=PROTEIN -INFILE={outfile}.aln -OUTFILE={outfile}.clust.aln >/dev/null 2>&1"
    os.system(cmd)

    return read_clust_aln_file(f"{outfile}.clust.aln")


def cal_identitcal_ratio(clustalw_program, casp_seq, pdb_seq, outfile):
    casp_aln, pdb_aln = align_sequence(clustalw_program, casp_seq, pdb_seq, outfile)
    return casp_aln, pdb_aln, len([i for i in range(len(casp_aln)) if casp_aln[i] != '-' and casp_aln[i] == pdb_aln[i]])

# s input str
# c search char
def find_all(s, c):
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)

def align_pdb_sequences_with_casp_sequences(clustalw_program, chain_pdbs, sequence_id_map):
    chain_align_matrix = []
    chain_ids = []
    for chain_pdb in chain_pdbs:
        chain_sequence = get_sequence(chain_pdb)
        identical_ratios = []
        for sequence in sequence_id_map:
            _, _, identical_ratio = cal_identitcal_ratio(clustalw_program, sequence, chain_sequence['sequence'], chain_pdb + '_chain' + sequence_id_map[sequence]['chain_id'])
            for _ in range(sequence_id_map[sequence]['count']):
                identical_ratios += [identical_ratio]
                chain_ids += [sequence_id_map[sequence]['chain_id']]
        chain_align_matrix += [identical_ratios]
    
    chain_align_matrix = np.array(chain_align_matrix)
    row_ind, col_ind = linear_sum_assignment(chain_align_matrix, maximize=True)
    return row_ind, col_ind, chain_ids


def get_chain_mapping(clustalw_program, sequence_id_map, inpdb, pdbdir):

    chain_pdbs_raw = split_pdb(inpdb, pdbdir)
    chain_pdbs = []
    for chain_pdb in chain_pdbs_raw:
        reindex_pdb_file(chain_pdb, chain_pdb + '.reindex')
        chain_pdbs += [chain_pdb + '.reindex']

    # get chain mapping from pdb to fasta file
    chain_mapping = {}
    
    pdb_indices, chain_ids_indices, chain_ids = align_pdb_sequences_with_casp_sequences(clustalw_program, chain_pdbs, sequence_id_map)
    
    for pdb_index, chain_index in zip(pdb_indices, chain_ids_indices):
        chain_pdb = chain_pdbs[pdb_index]
        pdb_sequence = get_sequence(chain_pdb)

        chain_id = chain_ids[chain_index]
        aln_file = chain_pdb + '_chain' + chain_id + '.clust.aln'
        casp_chain_aln, pdb_chain_aln = read_clust_aln_file(aln_file)

        pdb_indices_keep = []
        pdb_indices_order = []
        pdb_indices_counter = 0
        valid_char_counter = -1
        for i in range(len(casp_chain_aln)):

            if casp_chain_aln[i] != '-':
                pdb_indices_counter += 1

            if pdb_chain_aln[i] != '-':
                valid_char_counter += 1

            if casp_chain_aln[i] != '-' and pdb_chain_aln[i] != '-':
                pdb_indices_keep += [pdb_sequence['mapping'][valid_char_counter]]
                pdb_indices_order += [pdb_indices_counter]

        reindex_pdb_file(chain_pdb, chain_pdb + '.aligned', pdb_indices_keep, pdb_indices_order)

        chain_mapping[chain_pdb + '.aligned'] = chain_id

    # sort chain mapping by chain ids
    chain_mapping_sorted_by_chain_ids = sorted(chain_mapping.items(), key=lambda x:x[1])
    chain_mapping_sorted = dict(chain_mapping_sorted_by_chain_ids)

    return chain_mapping_sorted

def get_sequence_by_chain(chain_id, sequence_id_map):
    for sequence in sequence_id_map:
        if sequence_id_map[sequence]['chain_id'] == chain_id:
            return sequence
    return ''

def merge_chain_pdbs(chain_mapping, outfile):
    # reorder chains based on the stoichiometry
    # e.g., A2B2: AB CD
    chain_idx = 0
    with open(outfile, 'w') as fw:
        contents = []
        for chain_pdb in chain_mapping:
            for line in open(chain_pdb):
                if line.startswith('ATOM'):
                    line = line.rstrip('\n')
                    contents += [line[:21] + PDB_CHAIN_IDS[chain_idx] + line[22:]]
            contents += ["TER"]
            chain_idx += 1
        #print(contents)
        contents.pop(len(contents)-1)
        contents += ["END"]
        #print(contents)
        fw.write('\n'.join(contents))


def filter_single_model(inparams):
    
    clustalw_program, sequence_id_map, inpdb, pdbdir, outpdb = inparams

    # print(f"Filtering {inpdb}")

    # get chain mapping from pdb to fasta file
    chain_mapping = get_chain_mapping(clustalw_program=clustalw_program,
                                      sequence_id_map=sequence_id_map, 
                                      inpdb=inpdb,
                                      pdbdir=pdbdir)
    # print(chain_mapping)
    merge_chain_pdbs(chain_mapping, outpdb)

    os.system(f"rm -rf {pdbdir}")


def align_models(clustalw_program, fasta_path, outdir, input_model_dir):

    # read sequences from fasta file
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
    
    unique_sequences = list(sequence_id_map.keys())
    chain_ids = [sequence_id_map[sequence]['chain_id'] for sequence in sequence_id_map]

    makedir_if_not_exists(outdir)

    models_num = len(os.listdir(input_model_dir))

    process_list = []
    for model in sorted(os.listdir(input_model_dir)):

        workdir = outdir + '/' + model + '_temp'

        makedir_if_not_exists(workdir)

        process_list.append([clustalw_program, sequence_id_map, input_model_dir + '/' + model, workdir, outdir + '/' + model.replace('.pdb', '')])

    pool = Pool(processes=40)
    results = pool.map(filter_single_model, process_list)
    pool.close()
    pool.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--clustalw_program', type=str, required=True)
    
    args = parser.parse_args()

    align_models(args.clustalw_program, args.fasta_path, args.outdir, args.modeldir)

