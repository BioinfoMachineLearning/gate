import os, sys, argparse, time
import numpy as np
import pandas as pd
from collections import defaultdict

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
    chain_plddts = {}
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
                        chain_plddts[record['chain']].append(float(record['B']))

                else:
                    pdb_chains[record['chain']] = [line]
                    chain_coords[record['chain']]= [[record['x'],record['y'],record['z']]]
                    chain_CA_inds[record['chain']]= []
                    chain_CB_inds[record['chain']]= []
                    chain_plddts[record['chain']]= []
                    #Reset coord ind
                    coord_ind = 0

    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds, chain_plddts

def get_avg_interface_plddt(path_coords, path_CB_inds, path_plddt):
    '''
    Score all interfaces in the current complex

    Modified from the score_complex() function in MoLPC repo: 
    https://gitlab.com/patrickbryant1/molpc/-/blob/main/src/complex_assembly/score_entire_complex.py#L106-154
    '''

    chains = [*path_coords.keys()]
    chain_inds = np.arange(len(chains))
    sum_av_if_plDDT = 0
    n_chain_ints = 0
    #Get interfaces per chain
    for i in chain_inds:
        chain_i = chains[i]
        chain_coords = np.array(path_coords[chain_i])
        chain_CB_inds = path_CB_inds[chain_i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = path_plddt[chain_i]
        #print(chain_plddt)
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
                sum_av_if_plDDT += np.concatenate((np.array(chain_plddt)[contacts[:,0]], np.array(int_chain_plddt)[contacts[:,1]])).mean()
                n_chain_ints += 1

    if n_chain_ints == 0:
        print(sum_av_if_plDDT)
        return sum_av_if_plDDT

    return sum_av_if_plDDT / n_chain_ints

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    
    args = parser.parse_args()

    predictiors = ["no_esm_pred_pdbs", "no_plddt_pred_pdbs", "esm_plddt_pred_pdbs"]

    for predictor in predictiors:
        outdir = args.outdir + '/' + predictor
        os.makedirs(outdir, exist_ok=True)
        print(f"Processing {predictor}")
        for target in os.listdir(args.indir):
            print(f"Processing {target}")
            data_dict = {'model': [], 'iplddt': []}
            if len(os.listdir(args.modeldir + '/' + target)) != len(os.listdir(args.indir + '/' + target + '/' + predictor)):
                raise Exception(f"The model number is not consistent with the input directory : {target}")

            for inpdb in os.listdir(args.indir + '/' + target + '/' + predictor):
                pdb_path = args.indir + '/' + target + '/' + predictor + '/' + inpdb
                pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds, chain_plddts = read_pdb(pdb_path)
                avg_if_plddt = get_avg_interface_plddt(chain_coords,chain_CB_inds,chain_plddts)
                data_dict['model'] += [inpdb]
                data_dict['iplddt'] += [avg_if_plddt/100]
            df = pd.DataFrame(data_dict)
            df = df.sort_values(by='iplddt',ascending=False)
            df.to_csv(outdir + '/' + target + '.csv',index=False)