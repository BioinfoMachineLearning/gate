import os, sys, argparse
from typing import Optional
import numpy as np
import pandas as pd

def makedir_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory


def read_qa_txt_as_df(infile):
    models = []
    scores = []
    for line in open(infile):
        line = line.rstrip('\n')
        if len(line) == 0:
            continue
        contents = line.split()
        if contents[0] == "PFRMAT" or contents[0] == "TARGET" or contents[0] == "MODEL" or contents[0] == "QMODE" or \
                contents[0] == "END" or contents[0] == "REMARK":
            continue

        model = contents[0]

        if model.find('/') >= 0:
            model = os.path.basename(model)

        score = contents[1]

        models += [model]
        scores += [float(score)]

    df = pd.DataFrame({'model': models, 'score': scores})
    df = df.sort_values(by=['score'], ascending=False)
    df.reset_index(inplace=True)
    return df

def generate_feature_summary_ts(workdir, gate_prediction_df):

    QA_scores = ['af_plddt_avg', 'pairwise_gdt', 'pairwise_tmscore', 'pairwise_cad_score', 
                 'pairwise_lddt', 'enqa', 'gcpnet_ema', 'deeprank3_cluster', 'deeprank3_singleqa', 'deeprank3_singleqa_lite']

    af_plddt_avg_dict = {}
    enqa_dict = {}
    gcpnet_esm_plddt_dict = {}
    pairwise_gdt_dict, pairwise_tmscore_dict, pairwise_cad_score_dict, pairwise_lddt_dict = {}, {}, {}, {}
    deeprank3_cluster_dict, deeprank3_singleqa_dict, deeprank3_singleqa_lite_dict = {}, {}, {}
    models_for_targets = []
    for QA_score in QA_scores:
        if QA_score == 'gcpnet_ema':
            csv_file = os.path.join(workdir, 'gcpnet_ema', 'esm_plddt.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score']
                gcpnet_esm_plddt_dict[model] = score  
        elif QA_score == 'af_plddt_avg':
            csv_file = os.path.join(workdir, 'plddt', 'plddt.csv')
            df = pd.read_csv(csv_file)
            models_for_targets = df['model']
            for i in range(len(df)):
                model = df.loc[i, 'model']
                plddt = df.loc[i, 'plddt']
                af_plddt_avg_dict[model] = plddt
        elif QA_score == "enqa":
            csv_file = os.path.join(workdir, 'enqa', 'enqa.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score']
                enqa_dict[model] = score
        elif QA_score == "pairwise_gdt":
            csv_file = os.path.join(workdir, 'tmscore_pairwise', 'pairwise_gdtscore.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                gdtscores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_gdt_dict[model] = np.mean(np.array(gdtscores))
        elif QA_score == "pairwise_tmscore":
            csv_file = os.path.join(workdir, 'tmscore_pairwise', 'pairwise_tmscore.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                tmscores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_tmscore_dict[model] = np.mean(np.array(tmscores))
        elif QA_score == "pairwise_cad_score":
            csv_file = os.path.join(workdir, 'interface_pairwise', 'cad_score.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                cad_scores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_cad_score_dict[model] = np.mean(np.array(cad_scores))
        elif QA_score == "pairwise_lddt":
            csv_file = os.path.join(workdir, 'interface_pairwise', 'lddt.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                lddt_scores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_lddt_dict[model] = np.mean(np.array(lddt_scores))

        elif QA_score == "deeprank3_cluster":
            csv_file = os.path.join(workdir, 'DeepRank3', 'DeepRank3_Cluster.txt')
            df = read_qa_txt_as_df(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score']
                deeprank3_cluster_dict[model] = score

        elif QA_score == "deeprank3_singleqa":
            csv_file = os.path.join(workdir, 'DeepRank3', 'DeepRank3_SingleQA.txt')
            df = read_qa_txt_as_df(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score']
                deeprank3_singleqa_dict[model] = score

        elif QA_score == "deeprank3_singleqa_lite":
            csv_file = os.path.join(workdir, 'DeepRank3', 'DeepRank3_SingleQA_lite.txt')
            df = read_qa_txt_as_df(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score']
                deeprank3_singleqa_lite_dict[model] = score            

    gate_score_dict = {gate_prediction_df.loc[i, 'model']: gate_prediction_df.loc[i, 'score'] for i in range(len(gate_prediction_df))}

    data_dict = {'model': [], 'af_plddt_avg': [],
                'pairwise_gdt': [], 'pairwise_tmscore': [], 'pairwise_cad_score': [], 'pairwise_lddt': [],
                'enqa': [], 'gcpnet_ema': [],
                'deeprank3_cluster': [], 'deeprank3_singleqa': [], 'deeprank3_singleqa_lite': [],
                'gate': []}
    
    for model in models_for_targets:
        data_dict['model'] += [model]
        data_dict['af_plddt_avg'] += [af_plddt_avg_dict[model]]
        data_dict['pairwise_gdt'] += [pairwise_gdt_dict[model]]
        data_dict['pairwise_tmscore'] += [pairwise_tmscore_dict[model]]
        data_dict['pairwise_cad_score'] += [pairwise_cad_score_dict[model]]
        data_dict['pairwise_lddt'] += [pairwise_lddt_dict[model]]
        data_dict['enqa'] += [enqa_dict[model]]
        data_dict['gcpnet_ema'] += [gcpnet_esm_plddt_dict[model]]
        data_dict['deeprank3_cluster'] += [deeprank3_cluster_dict[model]]
        data_dict['deeprank3_singleqa'] += [deeprank3_singleqa_dict[model]]
        data_dict['deeprank3_singleqa_lite'] += [deeprank3_singleqa_lite_dict[model]]
        data_dict['gate'] += [gate_score_dict[model]]
    
    summary_df = pd.DataFrame(data_dict)
    return summary_df

def generate_feature_summary(workdir, gate_prediction_df, use_af_feature):

    QA_scores = ['af_plddt_avg', 'contact', 'dproqa', 'pairwise', 'pairwise_aligned', 'pairwise_usalign', 
                 'pairwise_qsscore', 'interface_pairwise', 'voro_scores', 'enqa', 
                 'gcpnet_ema']

    if use_af_feature:
        QA_scores += ['af_features']

    af_plddt_avg_norm_dict = {}
    af_confidence_dict, af_iptm_dict, af_num_inter_paes_dict, af_pdockq_dict = {}, {}, {}, {}
    icps_dict, recall_dict = {}, {}
    dproqa_norm_dict = {}
    enqa_norm_dict = {}
    pairwise_dict, pairwise_aligned_dict, pairwise_usalign_dict, pairwise_qsscore_dict = {}, {}, {}, {}
    pairwise_cad_score_dict, pairwise_dockq_ave_dict, pairwise_dockq_wave_dict = {}, {}, {}
    GNN_sum_score_norm_dict, GNN_pcadscore_norm_dict, voromqa_dark_norm_dict = {}, {}, {}
    gcpnet_esm_plddt_norm_dict = {}
    models_for_targets = []
    for QA_score in QA_scores:
        if QA_score == "contact":
            csv_file = os.path.join(workdir, 'icps', 'icps.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                icps = df.loc[i, 'icps']
                recall = df.loc[i, 'recall']
                icps_dict[model] = icps
                recall_dict[model] = recall
        elif QA_score == "dproqa":
            csv_file = os.path.join(workdir, 'dproqa', 'dproqa.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                dockq_norm = df.loc[i, 'DockQ_norm']
                dproqa_norm_dict[model] = dockq_norm
        elif QA_score == 'gcpnet_ema':
            csv_file = os.path.join(workdir, 'gcpnet_ema', 'esm_plddt.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score_norm']
                gcpnet_esm_plddt_norm_dict[model] = score  
        elif QA_score == 'af_plddt_avg':
            csv_file = os.path.join(workdir, 'plddt', 'plddt.csv')
            df = pd.read_csv(csv_file)
            models_for_targets = df['model']
            for i in range(len(df)):
                model = df.loc[i, 'model']
                plddt = df.loc[i, 'plddt_norm']
                af_plddt_avg_norm_dict[model] = plddt
        elif QA_score == "enqa":
            csv_file = os.path.join(workdir, 'enqa', 'enqa.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                score = df.loc[i, 'score_norm']
                enqa_norm_dict[model] = score
        elif QA_score == "pairwise_aligned":
            csv_file = os.path.join(workdir, 'mmalign_pairwise', 'pairwise_mmalign.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                scores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_aligned_dict[model] = np.mean(np.array(scores))
        elif QA_score == "pairwise_usalign":
            csv_file = os.path.join(workdir, 'usalign_pairwise', 'pairwise_usalign.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                scores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_usalign_dict[model] = np.mean(np.array(scores))
        elif QA_score == "pairwise_qsscore":
            csv_file = os.path.join(workdir, 'qsscore_pairwise', 'qsscore.csv')
            df = pd.read_csv(csv_file, index_col=[0])
            for model_idx, model in enumerate(df.columns):
                scores = [df[model][i] for i in range(len(df[model])) if i != model_idx]
                pairwise_qsscore_dict[model] = np.mean(np.array(scores))
        elif QA_score == "voro_scores":
            csv_file = os.path.join(workdir, 'voro', 'voro.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'model']
                GNN_sum_score_norm = df.loc[i, 'GNN_sum_score_norm']
                GNN_pcadscore_norm = df.loc[i, 'GNN_pcadscore_norm']
                voromqa_dark_norm = df.loc[i, 'voromqa_dark_norm']
                GNN_sum_score_norm_dict[model] = GNN_sum_score_norm
                GNN_pcadscore_norm_dict[model] = GNN_pcadscore_norm
                voromqa_dark_norm_dict[model] = voromqa_dark_norm
        elif QA_score == "af_features":
            csv_file = os.path.join(workdir, 'alphafold', 'af_features.csv')
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                model = df.loc[i, 'jobs']
                af_confidence_dict[model] = df.loc[i, 'iptm_ptm']
                af_iptm_dict[model] = df.loc[i, 'iptm']
                af_num_inter_paes_dict[model] = df.loc[i, 'num_inter_pae']
                af_pdockq_dict[model] = df.loc[i, 'mpDockQ/pDockQ']
        elif QA_score == 'interface_pairwise':
            for ema_score in ['cad_score', 'dockq_ave', 'dockq_wave']:
                csv_file = os.path.join(workdir, 'interface_pairwise', ema_score + '.csv')
                df = pd.read_csv(csv_file, index_col=[0])
                for model in df.columns:
                    if ema_score == 'cad_score':
                        pairwise_cad_score_dict[model] = np.mean(np.array(df[model]))
                    elif ema_score == 'dockq_ave':
                        pairwise_dockq_ave_dict[model] = np.mean(np.array(df[model]))
                    elif ema_score == 'dockq_wave':
                        pairwise_dockq_wave_dict[model] = np.mean(np.array(df[model]))
            continue
    
    gate_score_dict = {gate_prediction_df.loc[i, 'model']: gate_prediction_df.loc[i, 'score'] for i in range(len(gate_prediction_df))}

    data_dict = {'model': [], 
                'pairwise_aligned': [], 
                'pairwise_usalign': [], 'pairwise_qsscore': [],
                'pairwise_cad_score': [], 'pairwise_dockq_ave': [], 'pairwise_dockq_wave': [],
                'af_plddt_avg_norm': [],
                'icps': [], 'recall': [], 
                'dproqa_norm': [],
                'enqa_norm': [],           
                'GNN_sum_score_norm': [], 'GNN_pcadscore_norm': [], 'voromqa_dark_norm': [],
                'gcpnet_esm_plddt_norm': [],
                'gate': []}    

    if 'af_features' in QA_scores:
        data_dict['af_confidence'] = []
        data_dict['af_iptm'] = []
        data_dict['af_num_inter_pae'] = []
        data_dict['af_dockq'] = []

    for model in models_for_targets:
        data_dict['model'] += [model]
        data_dict['pairwise_aligned'] += [pairwise_aligned_dict[model]]
        data_dict['pairwise_usalign'] += [pairwise_usalign_dict[model]]
        data_dict['pairwise_qsscore'] += [pairwise_qsscore_dict[model]]

        data_dict['pairwise_cad_score'] += [pairwise_cad_score_dict[model]]
        data_dict['pairwise_dockq_ave'] += [pairwise_dockq_ave_dict[model]]
        data_dict['pairwise_dockq_wave'] += [pairwise_dockq_wave_dict[model]]

        data_dict['af_plddt_avg_norm'] += [af_plddt_avg_norm_dict[model]]
        data_dict['icps'] += [icps_dict[model]]
        data_dict['recall'] += [recall_dict[model]]
        data_dict['dproqa_norm'] += [dproqa_norm_dict[model]]
        data_dict['enqa_norm'] += [enqa_norm_dict[model]]
        data_dict['GNN_sum_score_norm'] += [GNN_sum_score_norm_dict[model]]
        data_dict['GNN_pcadscore_norm'] += [GNN_pcadscore_norm_dict[model]]
        data_dict['voromqa_dark_norm'] += [voromqa_dark_norm_dict[model]]
        data_dict['gcpnet_esm_plddt_norm'] += [gcpnet_esm_plddt_norm_dict[model]]

        data_dict['gate'] += [gate_score_dict[model]]
        
        if 'af_confidence' in data_dict:
            data_dict['af_confidence'] += [af_confidence_dict[model]]
            data_dict['af_iptm'] += [af_iptm_dict[model]]
            data_dict['af_num_inter_pae'] += [af_num_inter_paes_dict[model]]
            data_dict['af_dockq'] += [af_pdockq_dict[model]]

    summary_df = pd.DataFrame(data_dict)
    return summary_df