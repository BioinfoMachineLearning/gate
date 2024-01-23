import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    # workdirs = args.workdirs.split(',')
    #ensemble_predictions = sorted([infile for workdir in args.workdirs.split(',') for infile in os.listdir(workdir)])
    #ensemble_predictions = set(ensemble_predictions)
    #print(ensemble_predictions)
    workdirs = os.listdir(args.indir)
    ensemble_predictions = sorted([infile for workdir in workdirs for infile in os.listdir(args.indir + '/' + workdir) if infile != "DONE"])
    print(ensemble_predictions)

    for ensemble_prediction in ensemble_predictions:
        prev_df = None
        for i, workdir in enumerate(workdirs):
            
            if not os.path.exists(args.indir + '/' + workdir + '/' + ensemble_prediction):
                raise Exception(f"Cannot find {args.indir}/{workdir}/{ensemble_prediction}")
            
            curr_df = pd.read_csv(args.indir + '/' + workdir + '/' + ensemble_prediction)
            curr_df = curr_df.add_suffix(f"{i + 1}")
            curr_df['model'] = curr_df[f'model{i + 1}']
            curr_df = curr_df.drop([f'model{i + 1}'], axis=1)
            
            if prev_df is None:
                prev_df = curr_df
            else:
                prev_df = prev_df.merge(curr_df, on=f'model', how="inner")
        
        print(prev_df)
        avg_scores = []
        for i in range(len(prev_df)):
            sum_score = 0
            for j in range(len(workdirs)):
                sum_score += prev_df.loc[i, f'score{j+1}']
            avg_scores += [sum_score/len(workdirs)]
        
        models = prev_df['model']
        
        pd.DataFrame({'model': models, 'score': avg_scores}).to_csv(args.outdir + '/' + ensemble_prediction)
