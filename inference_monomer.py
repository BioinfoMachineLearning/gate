import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from gate.tool.utils import *
from gate.feature.feature_generation import *
from gate.sample.sample import *
from gate.feature.config import *
from gate.model.build_graph import *
from torch.utils.data import Dataset, DataLoader
from gate.model.graph_transformer_v3 import Gate


def cal_average_score(dfs):
    prev_df = None
    for i in range(len(dfs)):
        curr_df = dfs[i].add_suffix(f"{i + 1}")
        curr_df['model'] = curr_df[f'model{i + 1}']
        curr_df = curr_df.drop([f'model{i + 1}'], axis=1)
        if prev_df is None:
            prev_df = curr_df
        else:
            prev_df = prev_df.merge(curr_df, on=f'model', how="inner")
    
    # print(prev_df)
    avg_scores = []
    for i in range(len(prev_df)):
        sum_score = 0
        for j in range(len(dfs)):
            sum_score += prev_df.loc[i, f"score{j+1}"]

        avg_scores += [sum_score/len(dfs)]
    
    models = prev_df['model']
    
    ensemble_df = pd.DataFrame({'model': models, 'score': avg_scores})
    ensemble_df = ensemble_df.sort_values(by='score', ascending=False)
    ensemble_df.reset_index(inplace=True, drop=True)
    return ensemble_df

class DGLData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder):
        self.dgl_folder = dgl_folder
        self.data = []
        self.data_list = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_list[idx]

    def _prepare(self):
        for dgl_file in os.listdir(self.dgl_folder):
            g, tmp = dgl.data.utils.load_graphs(os.path.join(self.dgl_folder, dgl_file))
            self.data.append(g[0])
            self.data_list.append(os.path.join(self.dgl_folder, dgl_file))

def collate(samples):
    """Customer collate function"""
    graphs, data_paths = zip(*samples)
    batched_graphs = dgl.batch(graphs)
    return batched_graphs, data_paths

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--input_model_dir', type=str, required=True)
    parser.add_argument('--contact_map_file', type=str, required=True)
    parser.add_argument('--dist_map_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sample_times', default=5, type=int)

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    targetname = open(args.fasta_path).readlines()[0].rstrip('\n')[1:]

    features_monomer = features_monomer_dict()

    monomer_feature_generation(targetname=targetname,
                               fasta_path=args.fasta_path, 
                               input_model_dir=args.input_model_dir, 
                               output_dir=args.output_dir, 
                               contact_map_file=args.contact_map_file,
                               dist_map_file=args.dist_map_file,
                               config=CONFIG,
                               features_monomer=features_monomer)
    
    to_be_average_dfs = []

    for i in range(args.sample_times):

        sample_i_output_dir = os.path.join(args.output_dir, 'workdir' + str(i))
        os.makedirs(sample_i_output_dir, exist_ok=True)

        sample_dir = os.path.join(sample_i_output_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)

        print("Start to sample subgraphs......")
        if not os.path.exists(os.path.join(sample_i_output_dir, 'sample.done')):

            model_to_cluster = sample_models_by_kmeans_monomer(pairwise_gdtscore_file=features_monomer.pairwise_gdtscore, 
                                                            pairwise_tmscore_file=features_monomer.pairwise_tmscore, 
                                                            pairwise_cad_score_file=features_monomer.pairwise_cad_score, 
                                                            pairwise_lddt_file=features_monomer.pairwise_lddt,
                                                            sample_number_per_target=3000,
                                                            outdir=sample_dir)

            with open(os.path.join(sample_i_output_dir, 'cluster.txt'), 'w') as fw:
                for modelname in model_to_cluster:
                    fw.write(f"{modelname}\t{model_to_cluster[modelname]}\n")

            os.system(f"touch {os.path.join(sample_i_output_dir, 'sample.done')}")

        print("Generating dgl files for the subgraphs.....")

        dgl_dir = os.path.join(sample_i_output_dir, 'dgl')
        
        if not os.path.exists(os.path.join(sample_i_output_dir, 'dgl.done')):
            generate_monomer_dgls(targetname=targetname,
                                sample_dir=sample_dir,
                                dgl_dir=dgl_dir,
                                features_monomer=features_monomer,
                                sim_threshold=0.2)
            
            if len(os.listdir(dgl_dir)) == 3000:
                os.system(f"touch {os.path.join(sample_i_output_dir, 'dgl.done')}")
        
        print("Generating predictions for the subgraphs.....")

        prediction_dir = os.path.join(sample_i_output_dir, 'prediction')
        os.makedirs(prediction_dir, exist_ok=True)

        gate_model_names, dgl_dirs = [], []
        gate_model_names.append('casp15_inhouse_ts')
        dgl_dirs.append(dgl_dir)

        for gate_model_name, dgl_dir in zip(gate_model_names, dgl_dirs):
            
            print(dgl_dir)

            test_data = DGLData(dgl_folder=dgl_dir)
            
            prediction_dfs = []

            for fold in range(10):
                
                result_csv = os.path.join(prediction_dir, f'fold{fold}.csv')

                if os.path.exists(result_csv):
                    print(f"Prediction for fold{fold} has been generated!")
                    df = pd.read_csv(os.path.join(prediction_dir, f'fold{fold}.csv'))
                    prediction_dfs += [df]
                    continue

                fold_model_config = GATE_MODELS[gate_model_name]['fold' + str(fold)]

                test_loader = DataLoader(test_data,
                                        batch_size=fold_model_config.batch_size,
                                        num_workers=32,
                                        pin_memory=True,
                                        collate_fn=collate,
                                        shuffle=False)

                model = Gate(node_input_dim=fold_model_config.node_input_dim,
                                edge_input_dim=fold_model_config.edge_input_dim,
                                num_heads=fold_model_config.num_heads,
                                num_layer=fold_model_config.num_layer,
                                dp_rate=fold_model_config.dp_rate,
                                layer_norm=fold_model_config.layer_norm,
                                batch_norm=not fold_model_config.layer_norm,
                                residual=True,
                                hidden_dim=fold_model_config.hidden_dim,
                                mlp_dp_rate=fold_model_config.mlp_dp_rate)
                
                model = model.load_from_checkpoint(os.path.join(CKPTDIR, f"{gate_model_name}_fold{fold}.ckpt"))

                model = model.to(device)

                model.eval()

                target_pred_subgraph_scores = {}
                for idx, (batch_graphs, data_paths) in enumerate(test_loader):
                    batch_x = batch_graphs.ndata['f'].to(torch.float)
                    batch_e = batch_graphs.edata['f'].to(torch.float)
                    batch_graphs = batch_graphs.to(device)
                    batch_x = batch_x.to(device)
                    batch_e = batch_e.to(device)
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                    pred_scores = batch_scores.cpu().data.numpy().squeeze(1)

                    start_idx = 0
                    for subgraph_path in data_paths:
                        subgraph_filename = os.path.basename(subgraph_path)
                        subgraph_df = pd.read_csv(f"{sample_dir}/{subgraph_filename.replace('.dgl', '.csv')}", index_col=[0])
                        for i, modelname in enumerate(subgraph_df.columns):
                            if modelname not in target_pred_subgraph_scores:
                                target_pred_subgraph_scores[modelname] = []
                            target_pred_subgraph_scores[modelname] += [pred_scores[start_idx + i]]
                        start_idx += len(subgraph_df.columns)

                ensemble_scores, ensemble_count, std, normalized_std = [], [], [], []
                for modelname in target_pred_subgraph_scores:

                    mean_score = np.mean(np.array(target_pred_subgraph_scores[modelname]))
                    median_score = np.median(np.array(target_pred_subgraph_scores[modelname]))

                    if fold_model_config.ensemble_mode == "mean":
                        ensemble_scores += [mean_score]
                    else:
                        ensemble_scores += [median_score]
                        
                    ensemble_count += [len(target_pred_subgraph_scores[modelname])]
                    std += [np.std(np.array(target_pred_subgraph_scores[modelname]))]

                    normalized_std += [np.std(np.array(target_pred_subgraph_scores[modelname])) / mean_score]

                df = pd.DataFrame({'model': list(target_pred_subgraph_scores.keys()), 
                                'score': ensemble_scores, 
                                'sample_count': ensemble_count, 
                                'std': std, 
                                "std_norm": normalized_std})
                                
                df.to_csv(os.path.join(result_csv))

                prediction_dfs += [df]

            ensemble_df = cal_average_score(prediction_dfs)
            ensemble_df.to_csv(os.path.join(prediction_dir, 'ensemble.csv'))
            to_be_average_dfs += [ensemble_df]

    final_ensemble_df = cal_average_score(to_be_average_dfs)
    resultfile = os.path.join(args.output_dir, 'ensemble_af.csv')
    final_ensemble_df.to_csv(resultfile)

    # create a summary csv for all the scores in GATE
    summary_df = generate_feature_summary_ts(workdir=os.path.join(args.output_dir, 'feature'),
                                             gate_prediction_df=final_ensemble_df)
    summary_df.to_csv(os.path.join(args.output_dir, 'gate_af_summary.csv'))

if __name__ == '__main__':
    cli_main()

