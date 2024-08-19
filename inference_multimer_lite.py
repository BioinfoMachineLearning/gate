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
    parser.add_argument('--pkldir', type=str, default="", required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_af_feature', default=False, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    device = torch.device('cuda')  # set cuda device

    features_multimer = features_multimer_dict()

    complex_feature_generation_lite(fasta_path=args.fasta_path, 
                                    input_model_dir=args.input_model_dir, 
                                    output_dir=args.output_dir, 
                                    config=CONFIG, 
                                    use_alphafold_features=args.use_af_feature,
                                    pkldir=args.pkldir,
                                    features_multimer=features_multimer)

    
if __name__ == '__main__':
    cli_main()

