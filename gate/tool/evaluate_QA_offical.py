from calendar import c
from curses import raw
import os, sys, argparse, time
from pydoc import doc
from multiprocessing import Pool
from tqdm import tqdm
import random
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd

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
    parser.add_argument('--tarball', type=str, required=True)
    parser.add_argument('--nativedir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--groupid', type=str, required=False)
    parser.add_argument('--withscore', type=bool, default=False, required=False)
    args = parser.parse_args()

    group_ids = []
    
    if args.groupid is not None:
        group_ids = [args.groupid]
    else:
        for target in os.listdir(args.nativedir):
            target = target.rstrip('.csv')
            for prediction in os.listdir(args.tarball + '/' + target):
                # print(prediction)
                groupid = prediction[prediction.find('QA')+2:prediction.find('_')]
                if groupid not in group_ids:
                    group_ids += [groupid]

    group_ids = sorted(group_ids)
    
    group_res = {}
    for group_id in group_ids:
        # print(group_id)
        corrs = []
        best_tmscores = []
        losses = []
        max_tmscores = []
        global_qa = False
        for target in sorted(os.listdir(args.nativedir)):
            native_df = pd.read_csv(args.nativedir + '/' + target)
            # print(native_df)
            targetname = target.rstrip('.csv')

            scores_dict = {}
            for i in range(len(native_df)):
                scores_dict[native_df.loc[i, 'model']] = float(native_df.loc[i,'tmscore'])

            true_tmscores = native_df['tmscore']

            prediction = f"{args.tarball}/{targetname}/{targetname}QA{group_id}_1"
            print(prediction)
            if not os.path.exists(prediction):
                corrs += ["0"]
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
                continue
            
            # print(prediction)
            pred_df = read_qa_txt_as_df(targetname, prediction)
            # print(pred_df)
            if pred_df is None or len(pred_df) == 0:
                corrs += ["0"]
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
                continue

            scores_filt = []
            scores_true = []
            for i in range(len(pred_df)):
                model = pred_df.loc[i, 'model']
                if model not in scores_dict:
                    continue

                true_score = scores_dict[model]
                scores_filt += [pred_df.loc[i, 'score']]
                scores_true += [true_score]

            # print(scores_filt)
            # print(scores_true)

            if len(scores_filt) == 0:
                corrs += ["0"]
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
                continue

            corr = pearsonr(np.array(scores_filt), np.array(scores_true))[0]
            # print(corr)

            top1_models = [pred_df.loc[i, 'model'] for i in range(len(pred_df)) if float(pred_df.loc[i, 'score']) == float(pred_df.loc[0, 'score'])]
            if len(top1_models) == 0 or top1_models[0] not in scores_dict:
                corrs += ["0"]
                losses += [str(np.max(np.array(true_tmscores)))]
                best_tmscores += [np.max(np.array(true_tmscores))]
                max_tmscores += ["0"]
                continue
                # raise Exception(f"Cannot find the {scorefile} for {top1_model}!")

            # print(np.max(np.array(scores_true)))
            # print(true_scores_dict[top1_model])

            # best_top1_tmscore = np.max(np.array([float(scores_dict[top1_model]) for top1_model in top1_models]))
            best_top1_tmscore = float(scores_dict[top1_models[0]])
            loss = float(np.max(np.array(true_tmscores))) - best_top1_tmscore
            corrs += [str(corr)]
            losses += [str(loss)]
            best_tmscores += [str(float(np.max(np.array(scores_true))))]
            max_tmscores += [str(best_top1_tmscore)]

            global_qa = True

        if global_qa:
            group_res[group_id] = dict(corrs=corrs, losses=losses, best_tmscores=best_tmscores, select_tmscores=max_tmscores)

    group_ids = [key for key in group_res]
    print('  '.join(group_ids))
    
    targets = [target.rstrip('.csv') for target in sorted(os.listdir(args.nativedir))]
    print('\t'.join(targets))

    for i in range(len(os.listdir(args.nativedir))):
        contents = []
        for group_id in group_res:
            contents += [group_res[group_id]['corrs'][i]]
            contents += [group_res[group_id]['losses'][i]]
            if args.withscore:
                contents += [group_res[group_id]['best_tmscores'][i]]
                contents += [group_res[group_id]['select_tmscores'][i]]
        print('\t'.join(contents))




    
