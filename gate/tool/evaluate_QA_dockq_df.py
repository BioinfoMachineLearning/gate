from calendar import c
from curses import raw
import os, sys, argparse, time
from pydoc import doc
from multiprocessing import Pool
from tqdm import tqdm
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def read_qa_txt_as_df(targetname, infile):
    models = []
    scores = []
    for line in open(infile):
        line = line.rstrip('\n')
        # print(line)
        if len(line) == 0:
            continue
        contents = line.split()
        # if contents[0] == "QMODE":
        #     if float(contents[1]) == 2:
        #         return None

        if contents[0] == "PFRMAT" or contents[0] == "TARGET" or contents[0] == "MODEL" or contents[0] == "QMODE" or \
                contents[0] == "END" or contents[0] == "REMARK":
            continue

        contents = line.split()
        if contents[0].find(targetname) < 0:
            continue
        
        model, score1, score2 = contents[0], contents[1], contents[2]
        if score1 == "X":
            continue

        models += [model]
        scores += [float(score1)]
    df = pd.DataFrame({'model': models, 'score': scores})
    df = df.sort_values(by=['score'], ascending=False)
    df.reset_index(inplace=True)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--nativedir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--field', type=str, required=False)
    parser.add_argument('--ascending', type=bool, default=False, required=False)
    parser.add_argument('--nativefield', type=str, default='dockq_wave', required=False)

    args = parser.parse_args()

    scorefile = args.indir + '/' + os.listdir(args.indir)[0]
    if args.field is None:
        group_ids = pd.read_csv(scorefile).columns[2:]
    else:
        group_ids = [args.field]
    print(group_ids)
    group_res = {}
    for group_id in group_ids:
        # print(group_id)
        corrs = []
        spear_corrs = []
        best_scores = []
        losses = []
        MSEs = []
        for target in sorted(os.listdir(args.nativedir)):
            native_df = pd.read_csv(args.nativedir + '/' + target)
            # print(native_df)
            targetname = target.replace('.csv', '')

            scores_dict = {}
            for i in range(len(native_df)):
                scores_dict[native_df.loc[i, 'model']] = float(native_df.loc[i,args.nativefield])

            true_scores = native_df[args.nativefield]

            prediction = args.indir + '/' + target
            # print(prediction)
            
            # print(prediction)
            pred_df = pd.read_csv(prediction)
            # print(pred_df)
            if pred_df is None or len(pred_df) == 0:
                corrs += ["0"]
                spear_corrs += ["0"]
                losses += [str(np.max(np.array(true_scores)))]
                best_scores += [np.max(np.array(true_scores))]
                MSEs += ["-1"]
                continue

            pred_df = pred_df.sort_values(by=[group_id], ascending=args.ascending)
            pred_df.reset_index(inplace=True)
            # print(pred_df)

            scores_filt = []
            scores_true = []
            for i in range(len(pred_df)):
                model = pred_df.loc[i, 'model']
                if model not in scores_dict:
                    continue

                true_score = scores_dict[model]
                scores_filt += [float(pred_df.loc[i, group_id])]
                scores_true += [true_score]

            # print(scores_filt)
            # print(scores_true)

            if len(scores_filt) == 0:
                corrs += ["0"]
                spear_corrs += ["0"]
                losses += [str(np.max(np.array(true_scores)))]
                best_scores += [np.max(np.array(true_scores))]
                MSEs += ["-1"]
                continue

            # print(len(scores_dict))
            # print(len(scores_true))
            corr = pearsonr(np.array(scores_filt), np.array(scores_true))[0]
            spear_corr = spearmanr(np.array(scores_filt), np.array(scores_true)).statistic

            scaler = MinMaxScaler()
            if np.max(np.array(scores_filt)) > 1 or np.min(np.array(scores_filt)) < 0:
                scores_filt_norm = scaler.fit_transform(np.array(scores_filt).reshape(-1, 1))
                mse = mean_squared_error(scores_true, scores_filt_norm.reshape(-1))
            else:    
                mse = mean_squared_error(scores_true, scores_filt)
            # print(corr)

            top1_model = pred_df.loc[0, 'model']
            if top1_model not in scores_dict:
                corrs += ["0"]
                spear_corrs += ["0"]
                losses += [str(np.max(np.array(true_scores)))]
                best_scores += [np.max(np.array(true_scores))]
                MSEs += ["-1"]
                continue
                # raise Exception(f"Cannot find the {scorefile} for {top1_model}!")
            
            best_top1_score = float(scores_dict[top1_model])
            loss = float(np.max(np.array(true_scores))) - best_top1_score
            corrs += [str(corr)]
            spear_corrs += [str(spear_corr)]
            losses += [str(loss)]
            best_scores += [str(float(np.max(np.array(scores_true))))]
            MSEs += [str(mse)]

        group_res[group_id] = dict(corrs=corrs, spear_corrs=spear_corrs, losses=losses, 
                                   best_scores=best_scores, 
                                   MSEs=MSEs)

    group_ids = [key for key in group_res]
    print('    '.join(group_ids))
    
    targets = [target.rstrip('.csv') for target in sorted(os.listdir(args.nativedir))]
    print('\t'.join(targets))

    for i in range(len(os.listdir(args.nativedir))):
        contents = []
        for group_id in group_res:
            contents += [group_res[group_id]['corrs'][i]]
            contents += [group_res[group_id]['spear_corrs'][i]]
            contents += [group_res[group_id]['losses'][i]]
            contents += [group_res[group_id]['MSEs'][i]]
        print(' '.join(contents))




    
