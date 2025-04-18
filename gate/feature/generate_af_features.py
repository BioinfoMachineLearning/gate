#!/usr/bin/env python3

from math import pi
from operator import index
import os 
import pickle
import json
import numpy as np
import pandas as pd
import subprocess
import sys, argparse
from multiprocessing import Pool
import pathlib
from collections import defaultdict
import math

################FUNCTIONS#################
def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11].strip())
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26].strip())
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record

def read_pdb(pdbfile):
    '''Read a pdb file per chain
    '''
    pdb_chains = {}
    chain_coords = {}
    chain_CA_inds = {}
    chain_CB_inds = {}

    with open(pdbfile) as file:
        for line in file:
            if 'ATOM' in line:
                record = parse_atm_record(line)
                if record['chain'] in [*pdb_chains.keys()]:
                    pdb_chains[record['chain']].append(line)
                    chain_coords[record['chain']].append([record['x'],record['y'],record['z']])
                    coord_ind+=1
                    if record['atm_name']=='CA':
                        chain_CA_inds[record['chain']].append(coord_ind)
                    if record['atm_name']=='CB' or (record['atm_name']=='CA' and record['res_name']=='GLY'):
                        chain_CB_inds[record['chain']].append(coord_ind)


                else:
                    pdb_chains[record['chain']] = [line]
                    chain_coords[record['chain']]= [[record['x'],record['y'],record['z']]]
                    chain_CA_inds[record['chain']]= []
                    chain_CB_inds[record['chain']]= []
                    #Reset coord ind
                    coord_ind = 0


    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds

def get_best_plddt(work_dir):
    json_path = os.path.join(work_dir,'ranking_debug.json')
    best_model = json.load(open(json_path,'r'))['order'][0]
    best_plddt = pickle.load(open(os.path.join(work_dir,"result_{}.pkl".format(best_model)),'rb'))['plddt']
    
    return best_plddt

def read_plddt(best_plddt, chain_CA_inds):
    '''Get the plDDT for each chain
    '''
    chain_names = chain_CA_inds.keys()
    chain_lengths = dict()
    for name in chain_names:
        curr_len = len(chain_CA_inds[name])
        chain_lengths[name] = curr_len
    
    plddt_per_chain = dict()
    curr_len = 0
    for k,v in chain_lengths.items():
        curr_plddt = best_plddt[curr_len:curr_len+v]
        plddt_per_chain[k] = curr_plddt
        curr_len += v 
    return plddt_per_chain

def score_complex(path_coords, path_CB_inds, path_plddt):
    '''
    Score all interfaces in the current complex

    Modified from the score_complex() function in MoLPC repo: 
    https://gitlab.com/patrickbryant1/molpc/-/blob/main/src/complex_assembly/score_entire_complex.py#L106-154
    '''

    chains = [*path_coords.keys()]
    chain_inds = np.arange(len(chains))
    complex_score = 0
    #Get interfaces per chain
    for i in chain_inds:
        chain_i = chains[i]
        chain_coords = np.array(path_coords[chain_i])
        chain_CB_inds = path_CB_inds[chain_i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = path_plddt[chain_i]
 
        for int_i in np.setdiff1d(chain_inds, i):
            int_chain = chains[int_i]
            int_chain_CB_coords = np.array(path_coords[int_chain])[path_CB_inds[int_chain]]
            int_chain_plddt = path_plddt[int_chain]
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                av_if_plDDT = np.concatenate((chain_plddt[contacts[:,0]], int_chain_plddt[contacts[:,1]])).mean()
                complex_score += np.log10(contacts.shape[0]+1)*av_if_plDDT

    return complex_score, len(chains)

def calculate_mpDockQ(complex_score):
    """
    A function that returns a complex's mpDockQ score after 
    calculating complex_score
    """
    L = 0.827
    x_0 = 261.398
    k = 0.036
    b = 0.221
    return L/(1+math.exp(-1*k*(complex_score-x_0))) + b


def calc_pdockq(chain_coords, path_CB_inds, chain_plddt, t):
    '''Calculate the pDockQ scores
    pdockQ = L / (1 + np.exp(-k*(x-x0)))+b
    L= 0.724 x0= 152.611 k= 0.052 and b= 0.018

    Modified from the calc_pdockq() from FoldDock repo: 
    https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py#L62
    '''


    #Get coords and plddt per chain
    #ch1, ch2 = [*chain_coords.keys()]
    #coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    #plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    ch1, ch2 = [*path_CB_inds.keys()]
    coords1, coords2 = np.array(chain_coords[ch1])[path_CB_inds[ch1]], np.array(chain_coords[ch2])[path_CB_inds[ch2]]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]
    
    # print(plddt1)
    # print(plddt2)
    #Calc 2-norm
    mat = np.append(coords1, coords2,axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1,l1:] #upper triangular --> first dim = chain 1
    contacts = np.argwhere(contact_dists<=t)
    # print(contacts)
    if contacts.shape[0]<1:
        pdockq=0
    else:
        #Get the average interface plDDT
        avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:,0])], plddt2[np.unique(contacts[:,1])]]))
        #Get the number of interface contacts
        n_if_contacts = contacts.shape[0]
        x = avg_if_plddt*np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052*(x-152.611)))+0.018

    return pdockq

def examine_inter_pae(pae_mtx,seqs,cutoff):
    """A function that checks inter-pae values in multimer prediction jobs"""
    lens = [len(seq) for seq in seqs]
    old_lenth=0
    for length in lens:
        new_length = old_lenth + length
        pae_mtx[old_lenth:new_length,old_lenth:new_length] = 50
        old_lenth = new_length
    return np.where(pae_mtx<cutoff)[0].size


def obtain_mpdockq(pdb_path, result_dict):
    """Returns mpDockQ if more than two chains otherwise return pDockQ"""
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_path)
    best_plddt = result_dict['plddt'] #get_best_plddt(work_dir)
    plddt_per_chain = read_plddt(best_plddt,chain_CA_inds)
    #print(plddt_per_chain)
    # print(chain_coords)
    complex_score,num_chains = score_complex(chain_coords,chain_CB_inds,plddt_per_chain)
    if complex_score is not None and num_chains>2:
        mpDockq_or_pdockq = calculate_mpDockQ(complex_score)
    elif complex_score is not None and num_chains==2:
        mpDockq_or_pdockq = calc_pdockq(chain_coords,chain_CB_inds,plddt_per_chain,t=8)
    else:
        mpDockq_or_pdockq = "None"
    return mpDockq_or_pdockq

def get_feature(inparams):
    pdbfile, pklfile, seqs, cutoff = inparams
    print(f"now processing {pdbfile}")
    try:
        check_dict = pickle.load(open(pklfile,'rb'))
        # print(check_dict.keys())
        iptm_ptm_score = float(check_dict['ranking_confidence'])
        iptm_score = float(check_dict['iptm'])
        pae_mtx = check_dict['predicted_aligned_error']
        num_inter_pae = int(examine_inter_pae(pae_mtx,seqs,cutoff=cutoff))
        mpDockq_score = float(obtain_mpdockq(pdbfile, check_dict))
        score_dict = {'pdb': os.path.basename(pdbfile), 
                      'iptm_ptm_score': iptm_ptm_score, 
                      'iptm_score': iptm_score, 
                      'mpDockq_score': mpDockq_score, 
                      'num_inter_pae': num_inter_pae}
        return os.path.basename(pdbfile), iptm_ptm_score, iptm_score, mpDockq_score, num_inter_pae
    except Exception as e:
        print(e)
        return pdbfile, 0.0, 0.0, 0.0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Download pdb structure from pdb bank for monomer or dimer by list"
    parser.add_argument("-f", "--fasta_path", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-pkl", "--pkldir", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-pdb", "--pdbdir", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-o", "--outfile", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-c", "--cutoff", help="pdb name in lower case", type=float, default=5.0)
    parser.add_argument("-s", "--surface_thres", help="output folder for the pdb files", type=int, default=2)
    
    args = parser.parse_args()

    seqs = [line.rstrip('\n') for line in open(args.fasta_path) if line[0] != '>']

    pdbs = os.listdir(args.pdbdir)
    good_pdbs = []
    iptm_ptm = list()
    iptm = list()
    mpDockq_scores = list()
    num_inter_paes = list()
    process_list = []

    for pdb in pdbs:
        pklfile = os.path.join(args.pkldir, pdb + '.pkl')
        if not os.path.exists(pklfile):
            raise Exception(f"Cannot find the pickle file ({pklfile}) for {pdb}") 
        process_list.append([os.path.join(args.pdbdir, pdb), pklfile, seqs, args.cutoff])
    
    pool = Pool(processes=5)
    results = pool.map(get_feature, process_list)
    pool.close()
    pool.join()
    
    for result in results:
        pdb, iptm_ptm_score, iptm_score, mpDockq_score, num_inter_pae = result
        good_pdbs.append(pdb)
        iptm_ptm.append(iptm_ptm_score)
        iptm.append(iptm_score)
        mpDockq_scores.append(mpDockq_score)
        num_inter_paes.append(num_inter_pae)

    other_measurements_df=pd.DataFrame.from_dict({
        "jobs":good_pdbs,
        "iptm_ptm":iptm_ptm,
        "num_inter_pae": num_inter_paes,
        "iptm":iptm,
        "mpDockQ/pDockQ":mpDockq_scores
    })

    pi_score_df = other_measurements_df
    columns = list(pi_score_df.columns.values)
    columns.pop(columns.index('jobs'))
    pi_score_df = pi_score_df[['jobs'] + columns]
    pi_score_df = pi_score_df.sort_values(by='iptm_ptm',ascending=False)
    
    pi_score_df.to_csv(args.outfile,index=False)
    
