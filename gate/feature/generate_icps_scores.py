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

def run_msa_tool(inparams):
    msa_runner, input_fasta_path, msa_out_path = inparams
    msa_runner.query(input_fasta_path, msa_out_path)

class Chain:
    def __init__(self, sequence, count):
        self.sequence = sequence
        self.count = count

def run_cdpred_on_dimers(inparams):

    cdpred_env_path, cdpred_program_path, name1, name2, chain_pdbs, a3m, outdir = inparams

    os.chdir(cdpred_program_path)

    mode = "heterodimer"
    if len(chain_pdbs) == 1:
        mode = "homodimer"

    cmd = f"{cdpred_env_path}/bin/python lib/Model_predict.py -n {name1}_{name2} -p {' '.join(chain_pdbs)} -a {a3m} -m {mode} -o {outdir}"
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

            cmap_file = os.path.join(pdbdir, f"{pair}{len(model_contact_pairs[pair])}.cmap")
            dmap_file = os.path.join(pdbdir, f"{pair}{len(model_contact_pairs[pair])}.dmap")
            np.savetxt(cmap_file, cmap, fmt='%d')
            np.savetxt(dmap_file, dmap, fmt='%1.3f')

    return model_contact_pairs

def find_interaction_pairs_from_model(inparams):
    
    clustalw_program, sequence_id_map, inpdb, pdbdir = inparams

    print(f"Extracting interaction pairs from {inpdb}")

    model_contact_pairs = {}
    pair_json_file = os.path.join(pdbdir, 'pair.json')
    if not os.path.exists(pair_json_file):
        # get chain mapping from pdb to fasta file
        chain_mapping = get_chain_mapping(clustalw_program=clustalw_program,
                                          sequence_id_map=sequence_id_map, 
                                          inpdb=inpdb,
                                          pdbdir=pdbdir)

        # find contact pairs
        model_contact_pairs = get_contact_pairs(sequence_id_map=sequence_id_map,
                                                chain_mapping=chain_mapping, pdbdir=pdbdir)

        print(model_contact_pairs)
        with open(pair_json_file, 'w') as fw:
            json.dump(model_contact_pairs, fw, indent = 4)
    else:
        with open(pair_json_file) as f:
            model_contact_pairs = json.load(f)

    if len(model_contact_pairs) == 0:
        print(f"Cannot find any pairs for {inpdb}")

    return inpdb, model_contact_pairs


def get_icps_score(cdpred_cmap_file, model_cmap_file):
    cdpred_cmap = np.loadtxt(cdpred_cmap_file)
    model_cmap = np.loadtxt(model_cmap_file)
    prob_map = np.multiply(cdpred_cmap, model_cmap)
    indices = np.argwhere(prob_map > 0)
    if len(indices) > 0:
        prob_score = np.mean(np.array([prob_map[row_index, col_index] for row_index, col_index in indices]))
        return prob_score
    return 0.0

def get_recall_score(cdpred_cmap_file, model_cmap_file, prob_threshold=0.2):
    cdpred_cmap = np.loadtxt(cdpred_cmap_file)
    model_cmap = np.loadtxt(model_cmap_file)
    
    cdpred_cmap[cdpred_cmap < prob_threshold] = 0
    cdpred_cmap[cdpred_cmap >= prob_threshold] = 1

    prob_map = np.multiply(cdpred_cmap, model_cmap)
    
    return (prob_map > 0).sum() / (cdpred_cmap.shape[0] * cdpred_cmap.shape[1])


def generate_icps_scores(fasta_path, outdir, pairwise_score_csv, input_model_dir,
                         clustalw_program, cdpred_env_path, cdpred_program_path,
                         hhblits_binary_path, hhblits_databases, 
                         jackhmmer_binary, jackhmmer_database):

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

    model_dir = os.path.join(outdir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    all_contact_pairs = {}

    models_num = len(os.listdir(input_model_dir))

    process_list = []
    for model in sorted(os.listdir(input_model_dir)):

        workdir = os.path.join(model_dir, model)

        os.makedirs(workdir, exist_ok=True)

        chain_pdb_dir = os.path.join(workdir, 'monomer_pdbs')

        os.makedirs(chain_pdb_dir, exist_ok=True)

        process_list.append([clustalw_program, sequence_id_map, os.path.join(input_model_dir, model), chain_pdb_dir])

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
    fw1 = open(os.path.join(outdir, 'all_pairs.txt'), 'w')
    fw2 = open(os.path.join(outdir, 'valid_pairs.txt'), 'w')
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

    workdir = os.path.join(outdir, 'cdpred')
    os.makedirs(workdir, exist_ok=True)

    print("Start to generate monomer alignments...")

    msa_process_list = []
    for sequence in unique_sequences:
        if sequence_id_map[sequence]['chain_id'] not in chain_ids_to_process:
            continue

        msadir = f"{workdir}/{sequence_id_map[sequence]['chain_id']}"
        os.makedirs(msadir, exist_ok=True)

        monomer_fasta = msadir + '/' + sequence_id_map[sequence]['chain_id'] + '.fasta'
        with open(monomer_fasta, 'w') as fw:
            fw.write(f">{sequence_id_map[sequence]['chain_id']}\n")
            fw.write(f"{sequence}\n")

        a3mfile = monomer_fasta.replace('.fasta', '.a3m')
        if not os.path.exists(a3mfile) or len(open(a3mfile).readlines()) == 0:
            hhblits_runner = HHBlits(binary_path=hhblits_binary_path, databases=[hhblits_databases])
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
    reference_model_dir = workdir + '/refer_model'
    os.makedirs(reference_model_dir, exist_ok=True)

    pairwise_df = pd.read_csv(pairwise_score_csv, index_col=[0])
    models = pairwise_df.columns
    tmscores = np.array([np.mean(np.array(pairwise_df[model])) for model in models])
    # pairwise_df = pd.read_csv(pairwise_score_csv)
    # models = pairwise_df['model']
    # tmscores = pairwise_df['MMalign score']

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
                pdb_sequence = get_sequence(os.path.join(reference_model_dir, chain_model))
                if pdb_sequence['sequence'] == sequence:
                    chain_pdbs[sequence_id_map[sequence]['chain_id']] = os.path.join(reference_model_dir, chain_model)
                    break

        if len(chain_pdbs) == len(unique_sequences):
            os.system(f"touch {os.path.join(reference_model_dir, selected_model)}")
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
        os.makedirs(cdpred_dir, exist_ok=True)
        
        fullname = '_'.join([chain for chain in pair])
        cdpred_cmap_file = os.path.join(outdir, 'cdpred', '_'.join(list(pair)), 'predmap', f"{fullname}.htxt")
        if os.path.exists(cdpred_cmap_file):
            continue
        
        msadir = os.path.join(outdir, 'cdpred')
        if chain_id1 == chain_id2:
            a3m = os.path.join(msadir, chain_id1, f"{chain_id1}.a3m")
            run_cdpred_list.append([cdpred_env_path, cdpred_program_path, chain_id1, chain_id2, [chain_pdbs[chain_id1]], a3m, cdpred_dir])
        else:
            sto1 = os.path.join(msadir, chain_id1, f"{chain_id1}.sto")
            sto2 = os.path.join(msadir, chain_id2, f"{chain_id2}.sto")

            paired_a3m = pair_a3m(sto1, sto2, cdpred_dir)
            run_cdpred_list.append([cdpred_env_path, cdpred_program_path, chain_id1, chain_id2, [chain_pdbs[chain_id1], chain_pdbs[chain_id2]], paired_a3m, cdpred_dir])
                
    pool = Pool(processes=10)
    results = pool.map(run_cdpred_on_dimers, run_cdpred_list)
    pool.close()
    pool.join()

    # start to calculate icps score
    print("Start to calculate icps and recall score...")
    data_dict = {'model': [], 'icps': [], 'recall': []}
    for model in sorted(os.listdir(input_model_dir)):
        chain_dir = os.path.join(model_dir, model, 'monomer_pdbs' )
        icps_model_scores = []
        recall_model_scores = []
        for pair in valid_contact_pairs:
            cmap_files = read_files_by_prefix_and_ext(chain_dir, pair, 'cmap')
            # print(cmap_files)
            if len(cmap_files) == 0:
                continue
            
            fullname = '_'.join([chain for chain in pair])
            cdpred_cmap_file = os.path.join(outdir, 'cdpred', '_'.join(list(pair)), 'predmap', f"{fullname}.htxt")
            icps_model_pair_scores = []
            recall_model_pair_scores = []

            cal_list = [[cdpred_cmap_file, cmap_file] for cmap_file in cmap_files]
            # print(cal_list)
            pool = Pool(processes=150)
            results = pool.map(icps_recall_wrappeer, cal_list)
            pool.close()
            pool.join()

            for score1, score2 in results:
                icps_model_pair_scores += [score1]
                recall_model_pair_scores += [score2]

            icps_model_scores += [np.max(np.array(icps_model_pair_scores))]
            recall_model_scores += [np.max(np.array(recall_model_pair_scores))]
        
        data_dict['model'] += [model]
        # print(icps_model_scores)
        data_dict['icps'] += [np.sum(np.array(icps_model_scores)) / len(valid_contact_pairs)]
        data_dict['recall'] += [np.sum(np.array(recall_model_scores)) / len(valid_contact_pairs)]
    
    pd.DataFrame(data_dict).to_csv(os.path.join(outdir, 'icps.csv'))

def icps_recall_wrappeer(inparams):
    cdpred_cmap_file, cmap_file = inparams
    return get_icps_score(cdpred_cmap_file, cmap_file), get_recall_score(cdpred_cmap_file, cmap_file)


def generate_interface_model_size(fasta_path: str, 
                                  input_model_dir: str, 
                                  cdpreddir: str, 
                                  outfile: str):

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
        # print(aligned_pdb_files)
        for aligned_pdb_file in aligned_pdb_files:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('', aligned_pdb_file)
            chain_id = list(structure[0].child_dict.keys())
            xyzPDB = structure[0][chain_id[0]]
            aligned_model_size += len(xyzPDB)

        data_dict['model'] += [model]
        data_dict['interface_size_norm'] += [interface_size / target_length]
        data_dict['model_size_norm'] += [aligned_model_size / target_length]

    pd.DataFrame(data_dict).to_csv(outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--pairwise_score_csv', type=str, required=True)
    parser.add_argument('--hhblits_databases', type=str, required=True)
    parser.add_argument('--jackhmmer_database', type=str, required=True)
    parser.add_argument('--hhblits_binary_path', type=str, required=True)
    parser.add_argument('--jackhmmer_binary', type=str, required=True)
    parser.add_argument('--clustalw_program', type=str, required=True)
    parser.add_argument('--cdpred_env_path', type=str, required=True)
    parser.add_argument('--cdpred_program_path', type=str, required=True)

    args = parser.parse_args()

    generate_icps_scores(fasta_path=args.fasta_path, 
                         outdir=args.outdir, 
                         pairwise_score_csv=args.pairwise_score_csv, 
                         input_model_dir=args.model_dir,
                         clustalw_program=args.clustalw_program, 
                         cdpred_env_path=args.cdpred_env_path,
                         cdpred_program_path=args.cdpred_program_path,
                         hhblits_binary_path=args.hhblits_binary_path, 
                         hhblits_databases=args.hhblits_databases, 
                         jackhmmer_binary=args.jackhmmer_binary, 
                         jackhmmer_database=args.jackhmmer_database)

    outfile = os.path.join(args.outdir, 'model_size.csv')
    generate_interface_model_size(fasta_path=args.fasta_path,
                                  input_model_dir=args.model_dir,
                                  cdpreddir=args.outdir,
                                  outfile=outfile)
