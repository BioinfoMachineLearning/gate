import os, sys, argparse, time
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    QA_scores = ['alphafold', 'contact', 'dproqa', 'pairwise', 'voro_scores', 'enqa']
    targets = sorted(os.listdir(args.indir + '/' + QA_scores[0]))

    for target in targets:
        target = target.replace('.csv', '')
        af_plddt_avg_dict, af_plddt_avg_norm_dict = {}, {}
        icps_dict, recall_dict = {}, {}
        dproqa_dict, dproqa_norm_dict = {}, {}
        enqa_dict, enqa_norm_dict = {}, {}
        pairwise_dict = {}
        GNN_sum_score_dict, GNN_pcadscore_dict, voromqa_dark_dict = {}, {}, {}
        GNN_sum_score_norm_dict, GNN_pcadscore_norm_dict, voromqa_dark_norm_dict = {}, {}, {}

        models_for_targets = None
        for QA_score in QA_scores:
            csv_file = f"{args.indir}/{QA_score}/{target}.csv"
            df = pd.read_csv(csv_file)
            if QA_score == 'alphafold':
                models_for_targets = df['model']
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    plddt = df.loc[i, 'plddt']
                    plddt_norm = df.loc[i, 'plddt_norm']
                    af_plddt_avg_dict[model] = plddt
                    af_plddt_avg_norm_dict[model] = plddt_norm
            elif QA_score == "contact":
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    icps = df.loc[i, 'icps']
                    recall = df.loc[i, 'recall']
                    icps_dict[model] = icps
                    recall_dict[model] = recall
            elif QA_score == "dproqa":
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    dockq = df.loc[i, 'DockQ']
                    dockq_norm = df.loc[i, 'DockQ_norm']
                    dproqa_dict[model] = dockq
                    dproqa_norm_dict[model] = dockq_norm
            elif QA_score == "enqa":
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    score = df.loc[i, 'score']
                    score_norm = df.loc[i, 'score_norm']
                    enqa_dict[model] = score
                    enqa_norm_dict[model] = score_norm
            elif QA_score == "pairwise":
                df = pd.read_csv(csv_file, index_col=[0])
                for model in df.columns:
                    pairwise_dict[model] = np.mean(np.array(df[model]))
            elif QA_score == "voro_scores":
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    GNN_sum_score = df.loc[i, 'GNN_sum_score']
                    GNN_sum_score_norm = df.loc[i, 'GNN_sum_score_norm']
                    
                    GNN_pcadscore = df.loc[i, 'GNN_pcadscore']
                    GNN_pcadscore_norm = df.loc[i, 'GNN_pcadscore_norm']

                    voromqa_dark = df.loc[i, 'voromqa_dark']
                    voromqa_dark_norm = df.loc[i, 'voromqa_dark_norm']

                    GNN_sum_score_dict[model] = GNN_sum_score
                    GNN_sum_score_norm_dict[model] = GNN_sum_score_norm

                    GNN_pcadscore_dict[model] = GNN_pcadscore
                    GNN_pcadscore_norm_dict[model] = GNN_pcadscore_norm

                    voromqa_dark_dict[model] = voromqa_dark
                    voromqa_dark_norm_dict[model] = voromqa_dark_norm

        data_dict = {'model': [], 
                    'pairwise': [], 
                    'af_plddt_avg': [], 'af_plddt_avg_norm': [],
                    'icps': [], 'recall': [], 
                    'dproqa': [], 'dproqa_norm': [],
                    'enqa': [], 'enqa_norm': [],           
                    'GNN_sum_score': [], 'GNN_pcadscore': [], 'voromqa_dark': [],
                    'GNN_sum_score_norm': [], 'GNN_pcadscore_norm': [], 'voromqa_dark_norm': []}        
        
        for model in models_for_targets:
            data_dict['model'] += [model]
            data_dict['pairwise'] += [pairwise_dict[model]]
            data_dict['af_plddt_avg'] += [af_plddt_avg_dict[model]]
            data_dict['af_plddt_avg_norm'] += [af_plddt_avg_norm_dict[model]]
            data_dict['icps'] += [icps_dict[model]]
            data_dict['recall'] += [recall_dict[model]]
            data_dict['dproqa'] += [dproqa_dict[model]]
            data_dict['dproqa_norm'] += [dproqa_norm_dict[model]]
            data_dict['enqa'] += [enqa_dict[model]]
            data_dict['enqa_norm'] += [enqa_norm_dict[model]]
            data_dict['GNN_sum_score'] += [GNN_sum_score_dict[model]]
            data_dict['GNN_pcadscore'] += [GNN_pcadscore_dict[model]]
            data_dict['voromqa_dark'] += [voromqa_dark_dict[model]]
            data_dict['GNN_sum_score_norm'] += [GNN_sum_score_norm_dict[model]]
            data_dict['GNN_pcadscore_norm'] += [GNN_pcadscore_norm_dict[model]]
            data_dict['voromqa_dark_norm'] += [voromqa_dark_norm_dict[model]]
        
        pd.DataFrame(data_dict).to_csv(args.outdir + '/' + target + '.csv')

