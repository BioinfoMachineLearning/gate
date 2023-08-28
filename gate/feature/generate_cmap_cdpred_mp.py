import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
import itertools
from gate.tool.hhblits import HHBlits
from gate.tool.jackhmmer import Jackhmmer
from gate.tool.protein import *
from gate.tool.alignment import *
import copy
from scipy.optimize import linear_sum_assignment
import json

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

hhblits_databases = ['/home/bml_casp15/BML_CASP15/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt']
jackhmmer_database = '/home/bml_casp15/BML_CASP15/databases/uniref90/uniref90.fasta' 
hhblits_binary_path = '/home/bml_casp15/anaconda3/envs/bml_casp15/bin/hhblits'
jackhmmer_binary = '/home/bml_casp15/anaconda3/envs/bml_casp15/bin/jackhmmer'
clustalw_program = '/home/jl4mc/gate/tools/clustalw1.83/clustalw'
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


def read_clust_aln_file(infile):
    casp_align_seq = ""
    pdb_align_seq = ""
    for line in open(infile):
        if line[0:4] == "CASP":
            casp_align_seq += line.split()[1].rstrip('\n')
        if line[0:3] == "PDB":
            pdb_align_seq += line.split()[1].rstrip('\n')

    return casp_align_seq, pdb_align_seq


def align_sequence(casp_seq, pdb_seq, outfile):

    with open(outfile + '.aln', 'w') as fw: 
        fw.write(f"%CASP\n{casp_seq}\n%PDB\n{pdb_seq}")

    cmd = f"{clustalw_program}  -MATRIX=BLOSUM -TYPE=PROTEIN -INFILE={outfile}.aln -OUTFILE={outfile}.clust.aln >/dev/null 2>&1"
    os.system(cmd)

    return read_clust_aln_file(f"{outfile}.clust.aln")


def cal_identitcal_ratio(casp_seq, pdb_seq, outfile):
    casp_aln, pdb_aln = align_sequence(casp_seq, pdb_seq, outfile)
    return casp_aln, pdb_aln, len([i for i in range(len(casp_aln)) if casp_aln[i] != '-' and casp_aln[i] == pdb_aln[i]])

# s input str
# c search char
def find_all(s, c):
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)

def align_pdb_sequences_with_casp_sequences(chain_pdbs, sequence_id_map):
    chain_align_matrix = []
    chain_ids = []
    for chain_pdb in chain_pdbs:
        chain_sequence = get_sequence(chain_pdb)
        identical_ratios = []
        for sequence in sequence_id_map:
            _, _, identical_ratio = cal_identitcal_ratio(sequence, chain_sequence['sequence'], chain_pdb + '_chain' + sequence_id_map[sequence]['chain_id'])
            for _ in range(sequence_id_map[sequence]['count']):
                identical_ratios += [identical_ratio]
                chain_ids += [sequence_id_map[sequence]['chain_id']]
        chain_align_matrix += [identical_ratios]
    
    chain_align_matrix = np.array(chain_align_matrix)
    row_ind, col_ind = linear_sum_assignment(chain_align_matrix, maximize=True)
    return row_ind, col_ind, chain_ids


def get_chain_mapping(sequence_id_map, inpdb, pdbdir):

    chain_pdbs_raw = split_pdb(inpdb, pdbdir)
    chain_pdbs = []
    for chain_pdb in chain_pdbs_raw:
        reindex_pdb_file(chain_pdb, chain_pdb + '.reindex')
        chain_pdbs += [chain_pdb + '.reindex']

    # get chain mapping from pdb to fasta file
    chain_mapping = {}
    
    pdb_indices, chain_ids_indices, chain_ids = align_pdb_sequences_with_casp_sequences(chain_pdbs, sequence_id_map)
    
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

def get_contact_pairs(sequence_id_map, chain_mapping, pdbdir):
    
    model_contact_pairs = {}
    chain_pdbs = list(chain_mapping.keys())

    for i in range(len(chain_pdbs)):
        chain_pdb1 = chain_pdbs[i]
        for j in range(i+1, len(chain_pdbs)):
            chain_pdb2 = chain_pdbs[j]

            pair = chain_mapping[chain_pdb1] + chain_mapping[chain_pdb2]

            L1 = len(get_sequence_by_chain(chain_mapping[chain_pdb1], sequence_id_map))
            L2 = len(get_sequence_by_chain(chain_mapping[chain_pdb2], sequence_id_map))

            contact_num, cmap, dmap = cal_contact_number(pdb1=chain_pdb1, pdb2=chain_pdb2, L1=L1, L2=L2)

            if contact_num <= 0:
                continue

            if pair not in model_contact_pairs:
                model_contact_pairs[pair] = []
                
            # if (chain_pdb1, chain_pdb2) not in model_contact_pairs[pair]:    
            model_contact_pairs[pair] += [(chain_pdb1, chain_pdb2)]

            np.savetxt(f"{pdbdir}/{pair}{len(model_contact_pairs[pair])}.cmap", cmap, fmt='%d')
            np.savetxt(f"{pdbdir}/{pair}{len(model_contact_pairs[pair])}.dmap", dmap, fmt='%1.3f')

    return model_contact_pairs

def find_interaction_pairs_from_model(inparams):
    
    sequence_id_map, inpdb, pdbdir = inparams

    print(f"Extracting interaction pairs from {inpdb}")

    # get chain mapping from pdb to fasta file
    chain_mapping = get_chain_mapping(sequence_id_map=sequence_id_map, 
                                      inpdb=inpdb,
                                      pdbdir=pdbdir)

    # find contact pairs
    model_contact_pairs = get_contact_pairs(sequence_id_map=sequence_id_map,
                                            chain_mapping=chain_mapping, pdbdir=pdbdir)

    if len(model_contact_pairs) == 0:
        print(f"Cannot find any pairs for {inpdb}")
    else:
        print(model_contact_pairs)
        with open(pdbdir + '/pair.json', 'w') as fw:
            json.dump(model_contact_pairs, fw, indent = 4)

    return inpdb, model_contact_pairs


def get_icps_score(cdpred_cmap_file, model_cmap_file):
    cdpred_cmap = np.loadtxt(cdpred_cmap_file)
    model_cmap = np.loadtxt(model_cmap_file)
    prob_map = np.multiply(cdpred_cmap, model_cmap)
    indices = np.argwhere(prob_map > 0)
    prob_score = np.mean(np.array([prob_map[row_index, col_index] for row_index, col_index in indices]))
    return prob_score

def get_recall_score(cdpred_cmap_file, model_cmap_file, prob_threshold=0.2):
    cdpred_cmap = np.loadtxt(cdpred_cmap_file)
    model_cmap = np.loadtxt(model_cmap_file)
    
    cdpred_cmap[cdpred_cmap < prob_threshold] = 0
    cdpred_cmap[cdpred_cmap >= prob_threshold] = 1

    prob_map = np.multiply(cdpred_cmap, model_cmap)
    
    return (prob_map > 0).sum() / (cdpred_cmap.shape[0] * cdpred_cmap.shape[1])


def generate_icps_scores(targetname, fasta_path, outdir, pairwise_score_csv, input_model_dir, run_cdpred):

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

    model_dir = outdir + '/models'
    makedir_if_not_exists(model_dir)

    all_contact_pairs = {}

    models_num = len(os.listdir(input_model_dir))

    process_list = []
    for model in sorted(os.listdir(input_model_dir)):

        workdir = model_dir + '/' + model

        makedir_if_not_exists(workdir)

        chain_pdb_dir = workdir + '/monomer_pdbs' 

        makedir_if_not_exists(chain_pdb_dir)

        process_list.append([sequence_id_map, input_model_dir + '/' + model, chain_pdb_dir])

    pool = Pool(processes=40)
    results = pool.map(find_interaction_pairs_from_model, process_list)
    pool.close()
    pool.join()

    for result in results:
        modelname, model_contact_pairs = result
        for model_contact_pair in model_contact_pairs:
            if model_contact_pair not in all_contact_pairs:
                all_contact_pairs[model_contact_pair] = 1
            else:
                all_contact_pairs[model_contact_pair] += 1

    # Filter contact pairs based on the contact information from the model pool
    valid_contact_pairs = {}
    chain_ids_to_process = []
    fw1 = open(outdir + '/all_pairs.txt', 'w')
    fw2 = open(outdir + '/valid_pairs.txt', 'w')
    for contact_pair in all_contact_pairs:
        line = f"{contact_pair}: {all_contact_pairs[contact_pair]}"
        print(line)
        fw1.write(line)
        
        if all_contact_pairs[contact_pair] / models_num < 0.2:
            continue

        fw2.write(line)
        valid_contact_pairs[contact_pair] = copy.deepcopy(all_contact_pairs[contact_pair])
        for chain in contact_pair:
            if chain not in chain_ids_to_process:
                chain_ids_to_process += [chain]
    fw1.close()
    fw2.close()

    workdir = outdir + '/cdpred'
    makedir_if_not_exists(workdir)

    print("Start to generate monomer alignments...")

    msa_process_list = []
    for sequence in unique_sequences:
        if sequence_id_map[sequence]['chain_id'] not in chain_ids_to_process:
            continue

        msadir = f"{workdir}/{sequence_id_map[sequence]['chain_id']}"
        makedir_if_not_exists(msadir)

        monomer_fasta = msadir + '/' + sequence_id_map[sequence]['chain_id'] + '.fasta'
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

    if not run_cdpred:
        print("11111111111111111111111111111")
        return msa_process_list, []

    # pool = Pool(processes=15)
    # results = pool.map(run_msa_tool, msa_process_list)
    # pool.close()
    # pool.join()

    # find the model with the highest pairwise score - average similarity scores by column
    print("Searching the suitable model based on pairwise scores...")
    reference_model_dir = workdir + '/refer_model'
    makedir_if_not_exists(reference_model_dir)

    pairwise_df = pd.read_csv(pairwise_score_csv, index_col=[0])
    models = pairwise_df.columns
    tmscores = np.array([np.mean(np.array(pairwise_df[model])) for model in models])
    chain_pdbs = {}
    while True:
        select_model_idx = np.argmax(tmscores)
        selected_model = models[select_model_idx]
        print(f"Checking {selected_model}")

        os.system(f"rm {reference_model_dir}/*")
        os.system(f"cp {model_dir}/{selected_model}/monomer_pdbs/* {reference_model_dir}")

        chain_models = os.listdir(reference_model_dir)
        # need to pair the chain pdb and sequence
        for sequence in unique_sequences:
            for chain_model in chain_models:
                if chain_model.find('.pdb.reindex.aligned') < 0:
                    continue
                pdb_sequence = get_sequence(reference_model_dir + '/' + chain_model)
                if pdb_sequence['sequence'] == sequence:
                    chain_pdbs[sequence_id_map[sequence]['chain_id']] = reference_model_dir + '/' + chain_model
                    break

        if len(chain_pdbs) == len(unique_sequences):
            os.system(f"touch {reference_model_dir}/{selected_model}")
            print(f"Sucess!")
            break
        else:
            print(f"Cannot map the sequences: {len(chain_pdbs)} and {len(unique_sequences)}!")
            chain_pdbs = {}
            tmscores[select_model_idx] = 0.0
    
    # run cdpred on the valid dimer list
    print("Start to run CDPred...")
    run_cdpred_list = []
    for pair in valid_contact_pairs:
        print(pair)
        chain_id1, chain_id2 = pair[0], pair[1]
        cdpred_dir = f"{workdir}/{chain_id1}_{chain_id2}"
        makedir_if_not_exists(cdpred_dir)
        if os.path.exists(f"{cdpred_dir}/predmap/{targetname}{chain_id1}_{targetname}{chain_id2}.htxt"):
            continue
        
        msadir = outdir + '/cdpred'
        if chain_id1 == chain_id2:
            run_cdpred_list.append([targetname + chain_id1, targetname + chain_id2, [chain_pdbs[chain_id1]], f"{msadir}/{chain_id1}/{chain_id1}.a3m", cdpred_dir])
        else:
            paired_a3m = pair_a3m(f"{msadir}/{chain_id1}/{chain_id1}.sto", f"{msadir}/{chain_id2}/{chain_id2}.sto", cdpred_dir)
            run_cdpred_list.append([targetname + chain_id1, targetname + chain_id2, [chain_pdbs[chain_id1], chain_pdbs[chain_id2]], paired_a3m, cdpred_dir])
                
    # pool = Pool(processes=10)
    # results = pool.map(run_cdpred_on_dimers, run_cdpred_list)
    # pool.close()
    # pool.join()
    return msa_process_list, run_cdpred_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fastadir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--pairwise_dir', type=str, required=True)
    parser.add_argument('--run_cdpred',choices=('True','False'))

    args = parser.parse_args()

    run_cdpred = args.run_cdpred == 'True'
    msa_process_list = []
    run_cdpred_list = []
    for fastafile in sorted(os.listdir(args.fastadir)):
        targetname = fastafile.replace('.fasta', '')
        print(f"Processing {targetname}")

        pairwise_score_csv = args.pairwise_dir + '/' + targetname + '.csv'
        if not os.path.exists(pairwise_score_csv) and run_cdpred:
            print(f"Cannot find the pairwise score file: {pairwise_score_csv}")
            continue

        if not os.path.exists(args.modeldir + '/' + targetname):
            print(f"Cannot find the model directory: {args.modeldir}/{targetname}")
            continue

        outdir = args.outdir + '/' + targetname
        makedir_if_not_exists(outdir)

        target_msa_list, target_cdpred_list = generate_icps_scores(targetname, args.fastadir + '/' + fastafile, outdir, pairwise_score_csv, args.modeldir + '/' + targetname, run_cdpred)

        msa_process_list += target_msa_list
        run_cdpred_list += target_cdpred_list

    pool = Pool(processes=15)
    results = pool.map(run_msa_tool, msa_process_list)
    pool.close()
    pool.join()

    if len(run_cdpred_list) > 0:
        print(run_cdpred_list)
        pool = Pool(processes=10)
        results = pool.map(run_cdpred_on_dimers, run_cdpred_list)
        pool.close()
        pool.join()

