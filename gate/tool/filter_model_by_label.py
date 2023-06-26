import os, sys, argparse, time
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--labeldir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):
        outdir = args.outdir + '/' + target
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        label_file = args.labeldir + '/' + target + '.csv'
        if not os.path.exists(label_file):
            raise Exception(f"Cannot find label file: {label_file}")
        
        models_with_labels = list(pd.read_csv(label_file)['model'])

        for pdb in os.listdir(args.indir + '/' + target):
            if pdb in models_with_labels:
                os.system(f"cp {args.indir}/{target}/{pdb} {outdir}/{pdb}")
            else:
                print(f"{pdb} was filtered out because of missing label!")

