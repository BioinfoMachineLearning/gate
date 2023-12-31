import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        data = {'model': [], 'dockq_wave': [], 'dockq_ave': []}
        for pdb in os.listdir(args.indir + '/' + target):
            if pdb.find('.json') < 0:
                continue
            pdbname = pdb.replace('.pdb', '').replace('.json', '')
            json_file = args.indir + '/' + target + '/' + pdb
            print(json_file)
            data_json = json.load(open(json_file))
            dockq_wave = data_json['dockq_wave']
            dockq_ave = data_json['dockq_ave']
            data['model'] += [pdbname.replace('.pdb', '')]
            data['dockq_wave'] += [dockq_wave]
            data['dockq_ave'] += [dockq_ave]
        pd.DataFrame(data).to_csv(args.outdir + '/' + target + '.csv')
