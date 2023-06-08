import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from util import *
import re, subprocess
import itertools

af_program = "/home/bml_casp15/MULTICOM_dev/pipelines/default_proj/multimer_default.py"
option_file = "/home/bml_casp15/MULTICOM_dev/bin/db_option_default_af"

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

class Chain:
    def __init__(self, sequence, count):
        self.sequence = sequence
        self.count = count


def parse_fasta(fasta_string):
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions

def run_multicom3_default_af(infasta, outdir):
    cmd = f"python {af_program} --option_file {option_file} --fasta_path {infasta} --output_dir {outdir}"
    try:
        print(cmd)
        os.system(cmd)
    except Exception as e:
        print(e)


def generate_cmap_cdpred(targetname, fasta_path, outdir):

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
    unique_sequences = list(sequence_id_map.keys())
    dimers_sequences = []

    for i in range(len(unique_sequences)):
        sequence1 = unique_sequences[i]
        for j in range(i, len(unique_sequences)):
            sequence2 = unique_sequences[j]
            # check homodimer
            dimer_fasta, dimer_outdir = "", ""
            if sequence1 == sequence2:
                if sequence_id_map[sequence1]['count'] < 2:
                    continue
                workdir = f"{outdir}/{sequence_id_map[sequence1]['chain_id']}2"
                makedir_if_not_exists(workdir)

                dimer_fasta = workdir + '/' + targetname + '.fasta'
                dimer_outdir = workdir + '/alphafold'
                with open(dimer_fasta, 'w') as fw:
                    fw.write(f">{targetname}_A\n")
                    fw.write(f"{sequence1}\n")
                    fw.write(f">{targetname}_B\n")
                    fw.write(f"{sequence1}\n")
                
            else:
                workdir = f"{outdir}/{sequence_id_map[sequence1]['chain_id']}_{sequence_id_map[sequence2]['chain_id']}"
                makedir_if_not_exists(workdir)

                dimer_fasta = workdir + '/' + targetname + '.fasta'
                dimer_outdir = workdir + '/alphafold'
                with open(dimer_fasta, 'w') as fw:
                    fw.write(f">{targetname}_A\n")
                    fw.write(f"{sequence1}\n")
                    fw.write(f">{targetname}_B\n")
                    fw.write(f"{sequence2}\n")

            run_multicom3_default_af(dimer_fasta, dimer_outdir)
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fastadir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for fastafile in sorted(os.listdir(args.fastadir)):
        targetname = fastafile.replace('.fasta', '')
        print(f"Processing {targetname}")
        outdir = args.outdir + '/' + targetname
        makedir_if_not_exists(outdir)

        generate_cmap_cdpred(targetname, args.fastadir + '/' + fastafile, outdir)

