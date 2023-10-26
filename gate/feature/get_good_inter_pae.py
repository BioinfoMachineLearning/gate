#!/usr/bin/env python3

from math import pi
from operator import index
import os 
import pickle
import json
import numpy as np
import pandas as pd
import subprocess
from calculate_mpdockq import *
import sys, argparse
from multiprocessing import Pool
import pathlib

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

# def run_and_summarise_pi_score(workd_dir,jobs,surface_thres):

#     """A function to calculate all predicted models' pi_scores and make a pandas df of the results"""
#     try:
#         os.remove(f"mkdir {workd_dir}/pi_score_outputs")
#     except:
#         pass
#     subprocess.run(f"mkdir {workd_dir}/pi_score_outputs",shell=True,executable='/bin/bash')
#     pi_score_outputs = os.path.join(workd_dir,"pi_score_outputs")
#     for job in jobs:
#         subdir = os.path.join(workd_dir,job)
#         if not os.path.isfile(os.path.join(subdir,"ranked_0.pdb")):
#             print(f"{job} failed. Cannot find ranked_0.pdb in {subdir}")
#             sys.exit()
#         else:
#             pdb_path = os.path.join(subdir,"ranked_0.pdb")
#             output_dir = os.path.join(pi_score_outputs,f"{job}")
#             print(f"pi_score output for {job} will be stored at {output_dir}")
#             subprocess.run(f"source activate pi_score && export PYTHONPATH=/software:$PYTHONPATH && python /software/pi_score/run_piscore_wc.py -p {pdb_path} -o {output_dir} -s {surface_thres} -ps 10",shell=True,executable='/bin/bash')
            

#     output_df = pd.DataFrame()
#     for job in jobs:
#         subdir = os.path.join(pi_score_outputs,job)
#         csv_files = [f for f in os.listdir(subdir) if 'filter_intf_features' in f]
#         pi_score_files = [f for f in os.listdir(subdir) if 'pi_score_' in f]
#         filtered_df = pd.read_csv(os.path.join(subdir,csv_files[0]))
    
#         if filtered_df.shape[0]==0:
#             for column in filtered_df.columns:
#                 filtered_df[column] = ["None"]
#             filtered_df['jobs'] = str(job)
#             filtered_df['pi_score'] = "No interface detected"
#         else:
#             with open(os.path.join(subdir,pi_score_files[0]),'r') as f:
#                 lines = [l for l in f.readlines() if "#" not in l]
#                 if len(lines)>0:
#                     pi_score = pd.read_csv(os.path.join(subdir,pi_score_files[0]))
#                     pi_score['jobs']=str(job)
#                 else:
#                     pi_score = pd.DataFrame.from_dict({"pi_score":['SC:  mds: too many atoms']})
#                 f.close()
#             filtered_df['jobs'] = str(job)
#             filtered_df=pd.merge(filtered_df,pi_score,on='jobs')
#             try:
#                 filtered_df.drop(columns=["#PDB","pdb"," pvalue","chains","predicted_class"])
#             except:
#                 pass
        
#         output_df = pd.concat([output_df,filtered_df])
#     return output_df

def get_feature(inparams):
    pdbfile, pklfile, seqs, cutoff, outfile = inparams
    print(f"now processing {pdbfile}")
    try:
        check_dict = pickle.load(open(pklfile,'rb'))
        # print(check_dict.keys())
        iptm_ptm_score = float(check_dict['ranking_confidence'])
        iptm_score = float(check_dict['iptm'])
        pae_mtx = check_dict['predicted_aligned_error']
        num_inter_pae = int(examine_inter_pae(pae_mtx,seqs,cutoff=cutoff))
        mpDockq_score = float(obtain_mpdockq(pdbfile, check_dict))
        score_dict = {'pdb': pathlib.Path(pdbfile).stem, 
                      'iptm_ptm_score': iptm_ptm_score, 
                      'iptm_score': iptm_score, 
                      'mpDockq_score': mpDockq_score, 
                      'num_inter_pae': num_inter_pae}
        with open(outfile, 'w') as fw:
            json.dump(score_dict, fw, indent = 4)
        return pathlib.Path(pdbfile).stem, iptm_ptm_score, iptm_score, mpDockq_score, num_inter_pae
    except Exception as e:
        print(e)
        return pdbfile, 0.0, 0.0, 0.0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Download pdb structure from pdb bank for monomer or dimer by list"
    parser.add_argument("-f", "--fastadir", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-pkl", "--pkldir", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-pdb", "--pdbdir", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-o", "--output_dir", help="pdb name in lower case", type=str, required=True)
    parser.add_argument("-c", "--cutoff", help="pdb name in lower case", type=float, default=5.0)
    parser.add_argument("-s", "--surface_thres", help="output folder for the pdb files", type=int, default=2)
    
    args = parser.parse_args()

    for targetname in os.listdir(args.pdbdir):

        # if os.path.exists(args.output_dir + '/' + targetname + '.csv'):
        #     continue

        print(f"Processing {targetname}")
        outdir = os.path.join(args.output_dir, targetname)
        os.makedirs(outdir, exist_ok=True)

        fasta_path = args.fastadir + '/' + targetname + '.fasta'
        seqs = [line.rstrip('\n') for line in open(fasta_path) if line[0] != '>']

        pdbs = os.listdir(args.pdbdir + '/' + targetname)
        good_pdbs = []
        iptm_ptm = list()
        iptm = list()
        mpDockq_scores = list()
        num_inter_paes = list()
        process_list = []

        for pdb in pdbs:
            pklfile = args.pkldir + '/' + targetname + '/' + pdb + '.pkl'
            if not os.path.exists(pklfile):
                raise Exception(f"Cannot find the pickle file ({pklfile}) for {pdb}")
            json_file = os.path.join(outdir, pdb + '.json')
            if os.path.exists(json_file):
                with open(json_file) as f:
                    score_dict = json.load(f)

                    good_pdbs.append(score_dict['pdb'])
                    iptm_ptm.append(score_dict['iptm_ptm_score'])
                    iptm.append(score_dict['iptm_score'])
                    mpDockq_scores.append(score_dict['mpDockq_score'])
                    num_inter_paes.append(score_dict['num_inter_pae'])
            else:   
                process_list.append([args.pdbdir + '/' + targetname + '/' + pdb, pklfile, seqs, args.cutoff, json_file])
        
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
        #pi_score_df = run_and_summarise_pi_score(args.output_dir,good_jobs,args.surface_thres)
        #pi_score_df=pd.merge(pi_score_df,other_measurements_df,on="jobs")
        pi_score_df = other_measurements_df
        columns = list(pi_score_df.columns.values)
        columns.pop(columns.index('jobs'))
        pi_score_df = pi_score_df[['jobs'] + columns]
        pi_score_df = pi_score_df.sort_values(by='iptm_ptm',ascending=False)
        
        pi_score_df.to_csv(args.output_dir + '/' + targetname + '.csv',index=False)
    
