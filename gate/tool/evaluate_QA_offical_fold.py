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
    parser.add_argument('--folddir', type=str, required=True)
    parser.add_argument('--nativedir', type=str, required=True)
    
    args = parser.parse_args()

    for fold in range(10):
        folddir = f"{args.folddir}/fold{fold}"
        lines = open(folddir + '/targets.list').readlines()
        targets_test_in_fold = lines[2].split()
        print(f"Fold {fold}:")
        print(f"Test targets:")
        print(targets_test_in_fold)

        corrs = []
        spear_corrs = []
        best_tmscores = []
        losses = []
        max_tmscores = []
        MSEs = []

        for targetname in targets_test_in_fold:

            native_df = pd.read_csv(args.nativedir + '/' + targetname + '.csv')

            scores_dict = {}
            for i in range(len(native_df)):
                scores_dict[native_df.loc[i, 'model']] = float(native_df.loc[i,'tmscore'])

            true_tmscores = native_df['tmscore']

            prediction = args.indir + '/' + targetname + '.csv'
            # print(prediction)
            
            # print(prediction)
            pred_df = pd.read_csv(prediction)
            # print(pred_df)
            if pred_df is None or len(pred_df) == 0:
                corrs += ["0"]
                spear_corrs += ["0"]
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
                MSEs += ["-1"]
                continue

            pred_df = pred_df.sort_values(by=['score'], ascending=False)
            pred_df.reset_index(inplace=True)
            # print(pred_df)

            scores_filt = []
            scores_true = []
            for i in range(len(pred_df)):
                model = pred_df.loc[i, 'model']
                if model not in scores_dict:
                    continue

                true_score = scores_dict[model]
                scores_filt += [float(pred_df.loc[i, 'score'])]
                scores_true += [true_score]

            # print(scores_filt)
            # print(scores_true)

            if len(scores_filt) == 0:
                corrs += ["0"]
                spear_corrs += ["0"]
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
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
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
                MSEs += ["-1"]
                continue
                # raise Exception(f"Cannot find the {scorefile} for {top1_model}!")
            # print(top1_model)
            # print(np.max(np.array(true_tmscores)))
            # print(scores_dict[top1_model])

            # best_top1_tmscore = np.max(np.array([float(scores_dict[top1_model]) for top1_model in top1_models]))
            best_top1_tmscore = float(scores_dict[top1_model])
            loss = float(np.max(np.array(true_tmscores))) - best_top1_tmscore
            corrs += [corr]
            spear_corrs += [spear_corr]
            losses += [loss]
            best_tmscores += [float(np.max(np.array(scores_true)))]
            max_tmscores += [best_top1_tmscore]
            MSEs += [mse]

        for i in range(len(targets_test_in_fold)):
            contents = []
            contents += [str(corrs[i])]
            contents += [str(spear_corrs[i])]
            contents += [str(losses[i])]
            contents += [str(MSEs[i])]   
            print(' '.join(contents))
        print("Correlation\tSpear Correlation\tRanking loss\tMSE")
        print(f"Average: {np.mean(corrs)}\t{np.mean(spear_corrs)}\t{np.mean(losses)}\t{np.mean(MSEs)}")




    
