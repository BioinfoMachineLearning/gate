import os, sys, argparse, time
import numpy as np
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

        model = contents[0]

        if model.find('/') >= 0:
            model = os.path.basename(model)
            # print(model)

        score = contents[1]

        models += [model]
        scores += [float(score)]

    df = pd.DataFrame({'model': models, 'score': scores})
    df = df.sort_values(by=['score'], ascending=False)
    df.reset_index(inplace=True)
    # print(df)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    
    args = parser.parse_args()

    QA_scores = ['alphafold', 'pairwise_gdt', 'pairwise_tmscore', 'pairwise_cad_score', 
                 'pairwise_lddt', 'enqa', 'gcpnet_ema_pdb', 'deeprank3_cluster', 'deeprank3_singleqa', 'deeprank3_singleqa_lite']

    targets = sorted(os.listdir(args.indir + '/' + QA_scores[0]))

    for target in targets:
        if target.find('.csv') < 0:
            continue
        print("Processing " + target)
        target = target.replace('.csv', '')
        af_plddt_avg_dict = {}
        enqa_dict = {}
        gcpnet_esm_plddt_dict, gcpnet_noesm_dict, gcpnet_noplddt_dict = {}, {}, {}
        pairwise_gdt_dict, pairwise_tmscore_dict, pairwise_cad_score_dict, pairwise_lddt_dict = {}, {}, {}, {}
        deeprank3_cluster_dict, deeprank3_singleqa_dict, deeprank3_singleqa_lite_dict = {}, {}, {}

        models_for_targets = None
        for QA_score in QA_scores:
            if QA_score == 'gcpnet_ema_pdb':
                for ema_score in ['esm_plddt', 'no_esm', 'no_plddt']:
                    csv_file = f"{args.indir}/{QA_score}/{target}/{target}_{ema_score}.csv"
                    df = pd.read_csv(csv_file)
                    for i in range(len(df)):
                        model = df.loc[i, 'model']
                        score = df.loc[i, 'score']
                        if ema_score == 'esm_plddt':
                            gcpnet_esm_plddt_dict[model] = score
                        elif ema_score == 'no_esm':
                            gcpnet_noesm_dict[model] = score
                        elif ema_score == 'no_plddt':
                            gcpnet_noplddt_dict[model] = score
                continue

  
            if QA_score == 'alphafold':
                csv_file = f"{args.indir}/{QA_score}/{target}.csv"
                df = pd.read_csv(csv_file)
                models_for_targets = df['model']
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    plddt = df.loc[i, 'plddt']
                    af_plddt_avg_dict[model] = plddt
            elif QA_score == "enqa":
                csv_file = f"{args.indir}/{QA_score}/{target}.csv"
                df = pd.read_csv(csv_file)
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    score = df.loc[i, 'score']
                    enqa_dict[model] = score
            elif QA_score == "pairwise_gdt":
                csv_file = f"{args.indir}/pairwise/{target}_gdtscore.csv"
                df = pd.read_csv(csv_file, index_col=[0])
                for model in df.columns:
                    pairwise_gdt_dict[model] = np.mean(np.array(df[model]))
            elif QA_score == "pairwise_tmscore":
                csv_file = f"{args.indir}/pairwise/{target}_tmscore.csv"
                df = pd.read_csv(csv_file, index_col=[0])
                for model in df.columns:
                    pairwise_tmscore_dict[model] = np.mean(np.array(df[model]))

            elif QA_score == "pairwise_cad_score":
                csv_file = f"{args.indir}/interface_pairwise/{target}_cad_score.csv"
                df = pd.read_csv(csv_file, index_col=[0])
                for model in df.columns:
                    pairwise_cad_score_dict[model] = np.mean(np.array(df[model]))

            elif QA_score == "pairwise_lddt":
                csv_file = f"{args.indir}/interface_pairwise/{target}_lddt.csv"
                df = pd.read_csv(csv_file, index_col=[0])
                for model in df.columns:
                    pairwise_lddt_dict[model] = np.mean(np.array(df[model]))

            elif QA_score == "deeprank3_cluster":
                csv_file = f"{args.indir}/deeprank3/{target}/DeepRank3_Cluster.txt"
                df = read_qa_txt_as_df(target, csv_file)
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    score = df.loc[i, 'score']
                    deeprank3_cluster_dict[model] = score

            elif QA_score == "deeprank3_singleqa":
                csv_file = f"{args.indir}/deeprank3/{target}/DeepRank3_SingleQA.txt"
                df = read_qa_txt_as_df(target, csv_file)
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    score = df.loc[i, 'score']
                    deeprank3_singleqa_dict[model] = score

            elif QA_score == "deeprank3_singleqa_lite":
                csv_file = f"{args.indir}/deeprank3/{target}/DeepRank3_SingleQA_lite.txt"
                df = read_qa_txt_as_df(target, csv_file)
                for i in range(len(df)):
                    model = df.loc[i, 'model']
                    score = df.loc[i, 'score']
                    deeprank3_singleqa_lite_dict[model] = score                    
            

        data_dict = {'model': [], 'af_plddt_avg': [],
                    'pairwise_gdt': [], 'pairwise_tmscore': [], 'pairwise_cad_score': [], 'pairwise_lddt': [],
                    'enqa': [], 'gcpnet_esm_plddt': [], 'gcpnet_noesm': [], 'gcpnet_noplddt': [], 
                    'deeprank3_cluster': [], 'deeprank3_singleqa': [], 'deeprank3_singleqa_lite': []}
        
        for model in models_for_targets:
            data_dict['model'] += [model]
            data_dict['af_plddt_avg'] += [af_plddt_avg_dict[model]]
            data_dict['pairwise_gdt'] += [pairwise_gdt_dict[model]]
            data_dict['pairwise_tmscore'] += [pairwise_tmscore_dict[model]]
            data_dict['pairwise_cad_score'] += [pairwise_cad_score_dict[model]]
            data_dict['pairwise_lddt'] += [pairwise_lddt_dict[model]]
            data_dict['enqa'] += [enqa_dict[model]]
            data_dict['gcpnet_esm_plddt'] += [gcpnet_esm_plddt_dict[model]]
            data_dict['gcpnet_noesm'] += [gcpnet_noesm_dict[model]]
            data_dict['gcpnet_noplddt'] += [gcpnet_noplddt_dict[model]]

            data_dict['deeprank3_cluster'] += [deeprank3_cluster_dict[model]]
            data_dict['deeprank3_singleqa'] += [deeprank3_singleqa_dict[model]]
            data_dict['deeprank3_singleqa_lite'] += [deeprank3_singleqa_lite_dict[model]]
        
        pd.DataFrame(data_dict).to_csv(args.outdir + '/' + target + '.csv')

