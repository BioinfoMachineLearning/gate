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
    parser.add_argument('--pkldir', type=str, default="", required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_af_feature', default=False, type=bool, required=False)

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    features_multimer = features_multimer_dict()

    complex_feature_generation(fasta_path=args.fasta_path, 
                               input_model_dir=args.input_model_dir, 
                               output_dir=args.output_dir, 
                               config=CONFIG, 
                               use_alphafold_features=args.use_af_feature,
                               pkldir=args.pkldir,
                               features_multimer=features_multimer)

    sample_dir = os.path.join(args.output_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)

    print("Start to sample subgraphs......")
    if not os.path.exists(os.path.join(args.output_dir, 'sample.done')):
        model_to_cluster = sample_models_by_kmeans(pairwise_usalign_file=features_multimer.pairwise_usalign, 
                                                    pairwise_mmalign_file=features_multimer.pairwise_mmalign, 
                                                    pairwise_qsscore_file=features_multimer.pairwise_qsscore, 
                                                    pairwise_dockq_wave_file=features_multimer.pairwise_dockq_wave, 
                                                    pairwise_dockq_ave_file=features_multimer.pairwise_dockq_ave, 
                                                    pairwise_cad_score_file=features_multimer.pairwise_cad_score, 
                                                    sample_number_per_target=3000,
                                                    outdir=sample_dir)

        with open(os.path.join(args.output_dir, 'cluster.txt'), 'w') as fw:
            for modelname in model_to_cluster:
                fw.write(f"{modelname}\t{model_to_cluster[modelname]}\n")

        os.system(f"touch {os.path.join(args.output_dir, 'sample.done')}")

    print("Generating dgl files for the subgraphs.....")

    dgl_base_dir = os.path.join(args.output_dir, 'dgl')
    
    configs = {}

    if args.use_af_feature:
        configs = {
                    'v8': {'use_af_feature': args.use_af_feature,
                        'use_gcpnet_ema': True,
                        'use_interface_pairwise': True},
                  }
    else:
        configs = {
                    'v7_nogcpnet': {'use_af_feature': args.use_af_feature,
                                'use_gcpnet_ema': False,
                                'use_interface_pairwise': False},

                    'v7_gcpnet': {'use_af_feature': args.use_af_feature,
                                    'use_gcpnet_ema': True,
                                    'use_interface_pairwise': False},

                    'v8': {'use_af_feature': args.use_af_feature,
                            'use_gcpnet_ema': True,
                            'use_interface_pairwise': True},
                  }
    
    for config in configs:
        dgl_dir = os.path.join(dgl_base_dir, config)
        os.makedirs(dgl_dir, exist_ok=True)

        if os.path.exists(os.path.join(dgl_base_dir, config + '.done')):
            continue

        generate_multimer_dgls(sample_dir=sample_dir,
                               dgl_dir=dgl_dir,
                               features_multimer=features_multimer,
                               sim_threshold=0.5,
                               use_af_feature=configs[config]['use_af_feature'],
                               use_gcpnet_ema=configs[config]['use_gcpnet_ema'],
                               use_interface_pairwise=configs[config]['use_interface_pairwise'])
        
        if len(os.listdir(dgl_dir)) == 3000:
            os.system(f"touch {os.path.join(dgl_base_dir, config + '.done')}")
    
    print("Generating predictions for the subgraphs.....")

    prediction_base_dir = os.path.join(args.output_dir, 'prediction')
    
    gate_model_names, dgl_dirs = [], []

    if args.use_af_feature:
        # gate_model_names.append('casp15_inhouse_full_v7_nogcpnet')
        # dgl_dirs.append(os.path.join(dgl_base_dir, "v7_nogcpnet"))

        # gate_model_names.append('casp15_inhouse_full_v7_gcpnet')
        # dgl_dirs.append(os.path.join(dgl_base_dir, "v7_gcpnet"))

        # gate_model_names.append('casp15_inhouse_full_v8')
        # dgl_dirs.append(os.path.join(dgl_base_dir, "v8"))

        # gate_model_names.append('casp15_inhouse_top_v7_nogcpnet')
        # dgl_dirs.append(os.path.join(dgl_base_dir, "v7_nogcpnet"))

        # gate_model_names.append('casp15_inhouse_top_v7_gcpnet')
        # dgl_dirs.append(os.path.join(dgl_base_dir, "v7_gcpnet"))

        gate_model_names.append('casp15_inhouse_top_v8')
        dgl_dirs.append(os.path.join(dgl_base_dir, "v8"))

    else:
        gate_model_names.append('casp15_official_v7_nogcpnet')
        dgl_dirs.append(os.path.join(dgl_base_dir, "v7_nogcpnet"))

        gate_model_names.append('casp15_official_v7_gcpnet')
        dgl_dirs.append(os.path.join(dgl_base_dir, "v7_gcpnet"))

        gate_model_names.append('casp15_official_v8')
        dgl_dirs.append(os.path.join(dgl_base_dir, "v8"))

    ensemble_dfs = []

    for gate_model_name, dgl_dir in zip(gate_model_names, dgl_dirs):
        
        print(dgl_dir)

        test_data = DGLData(dgl_folder=dgl_dir)
        
        prediction_dir = os.path.join(prediction_base_dir, gate_model_name)
        os.makedirs(prediction_dir, exist_ok=True)

        prediction_dfs = []

        for fold in range(10):

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
            df.to_csv(os.path.join(prediction_dir, f'fold{fold}.csv'))

            prediction_dfs += [df]
    
        prev_df = None
        for i in range(10):
            curr_df = prediction_dfs[i].add_suffix(f"{i + 1}")
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
            for j in range(len(prediction_dfs)):
                sum_score += prev_df.loc[i, f"score{j+1}"]

            avg_scores += [sum_score/len(prediction_dfs)]
        
        models = prev_df['model']
        
        ensemble_df = pd.DataFrame({'model': models, 'score': avg_scores})
        ensemble_df.to_csv(os.path.join(prediction_dir, 'ensemble.csv'))

        ensemble_dfs += [ensemble_df]

    if not args.use_af_feature:

        prev_df = None
        for i in range(len(ensemble_dfs)):
            curr_df = ensemble_dfs[i].add_suffix(f"{i + 1}")
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
            for j in range(len(ensemble_dfs)):
                sum_score += prev_df.loc[i, f"score{j+1}"]

            avg_scores += [sum_score/len(ensemble_dfs)]
        
        models = prev_df['model']
        
        ensemble_df = pd.DataFrame({'model': models, 'score': avg_scores})
        ensemble_df.to_csv(os.path.join(args.output_dir, 'ensemble.csv'))


if __name__ == '__main__':
    cli_main()

