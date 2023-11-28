"""
@ Description: Protein to DGL graph with node and edge features
"""

import os, copy, time
import dgl
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Union
from joblib import Parallel, delayed
import pandas as pd
from graph_transformer_v2 import Gate
import lightning as L
from torch.utils.data import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger
import wandb
import scipy.sparse as sp
import torchmetrics
import json
import multiprocessing
from mango import scheduler, Tuner

os.environ["WANDB__SERVICE_WAIT"] = "3600"
os.environ["WANDB_API_KEY"] = "e84c57dee287170f97801b73a63280b155507e00"
torch.set_printoptions(profile="full")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def load_dgl_graph_and_label(inparams):
    index, dgl_file_path, label_file_path = inparams
    # Load the DGL graph from the file
    g, tmp = dgl.data.utils.load_graphs(dgl_file_path)
    label = np.load(label_file_path)
    return index, dgl_file_path, g, label

class DGLData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str, label_folder: str, targets):
        self.target_list = targets
        self.dgl_folder = dgl_folder
        self.label_folder = label_folder

        self.data_list = []
        self.data = []
        self.node_label = []
        # self.edge_label = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.node_label[idx], self.data_list[idx]#, self.edge_label[idx]

    def _prepare(self):
        for target in self.target_list:
            target_dgl_folder = self.dgl_folder + '/' + target
            target_label_folder = self.label_folder + '/' + target

            for dgl_file in os.listdir(target_dgl_folder):
                g, tmp = dgl.data.utils.load_graphs(target_dgl_folder + '/' + dgl_file)
                self.data.append(g[0])
                self.node_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_node.npy')))
                # self.edge_label.append(np.load(target_label_folder + '/' + dgl_file.replace('.dgl', '_edge.npy')))
                self.data_list.append(target_dgl_folder + '/' + dgl_file)

def collate(samples):
    """Customer collate function"""
    graphs, node_labels, data_paths = zip(*samples)
    batched_graphs = dgl.batch(graphs)
    batch_node_labels, batch_edge_labels = None, None
    for node_label in node_labels:
        if batch_node_labels is None:
            batch_node_labels = copy.deepcopy(node_label)
        else:
            batch_node_labels = np.concatenate((batch_node_labels, node_label), axis=None)
    
    return batched_graphs, torch.tensor(batch_node_labels).float().reshape(-1, 1), data_paths# , torch.tensor(batch_edge_labels).float().reshape(-1, 1)

def read_subgraph_columns(datadir, targets):
    subgraph_columns_dict = {}
    for targetname in os.listdir(datadir):
        if targetname not in targets:
            continue
        for subgraph in os.listdir(datadir + '/' + targetname):
            df = pd.read_csv(f"{datadir}/{targetname}/{subgraph}", index_col=[0])
            subgraph_columns_dict[f"{targetname}_{subgraph.replace('.csv', '')}"] = df.columns
    return subgraph_columns_dict

def read_native_dfs(labeldir, targets):
    native_dfs_dict = {}
    for infile in os.listdir(labeldir):
        targetname = infile.replace('.csv', '')
        if targetname in targets:
            df = pd.read_csv(labeldir + '/' + infile)
            native_dfs_dict[targetname] = df
    return native_dfs_dict

import pickle 

# Save pickle results
def save_res(data, file_name):
    pickle.dump( data, open( file_name, "wb" ) )

import time

# @scheduler.serial
def objfunc(args_list):

    objective_evaluated = []
    
    start_time = time.time()
    
    for hyper_par in args_list:
        objective = objective_graph_transformer(random_seed=hyper_par['random_seed'],
                                                projectname=hyper_par['projectname'], workdir=hyper_par['workdir'], 
                                                train_data=hyper_par['train_data'], val_data=hyper_par['val_data'],
                                                num_heads=hyper_par['num_heads'], 
                                                num_layer=hyper_par['num_layer'],
                                                dp_rate=hyper_par['dp_rate'],
                                                hidden_dim=hyper_par['hidden_dim'],
                                                mlp_dp_rate=hyper_par['mlp_dp_rate'],
                                                loss_fun=hyper_par['loss_fun'],
                                                lr=hyper_par['lr'],
                                                weight_decay=hyper_par['weight_decay'],
                                                layer_norm=hyper_par['layer_norm'],
                                                batch_size=hyper_par['batch_size'],
                                                ckpt_root_dir=hyper_par['ckpt_root_dir'],
                                                targets_train_in_fold=hyper_par['targets_train_in_fold'],
                                                targets_val_in_fold=hyper_par['targets_val_in_fold'],
                                                subgraph_columns_dict=hyper_par['subgraph_columns_dict'],
                                                native_dfs_dict=hyper_par['native_dfs_dict'],
                                                log_train_mse=hyper_par['log_train_mse'],
                                                log_val_mse=hyper_par['log_val_mse'])

        objective_evaluated.append(objective)
        
        end_time = time.time()
        print('objective:', objective, ' time:',end_time-start_time)
        
    return objective_evaluated

def objective_graph_transformer(random_seed, projectname, workdir, train_data, val_data, num_heads, num_layer, dp_rate, hidden_dim, 
                                mlp_dp_rate, loss_fun, lr, weight_decay, layer_norm,
                                batch_size, ckpt_root_dir, targets_train_in_fold, targets_val_in_fold,
                                subgraph_columns_dict, native_dfs_dict, log_train_mse, log_val_mse):

    node_input_dim = 18 # 20 #18
    edge_input_dim = 4 # 4 #3
    residual = True
    batch_norm = not layer_norm

    experiment_name = f"{node_input_dim}_" \
                        f"{edge_input_dim}_" \
                        f"{num_heads}_" \
                        f"{num_layer}_" \
                        f"{dp_rate}_" \
                        f"{layer_norm}_" \
                        f"{batch_norm}_" \
                        f"{residual}_" \
                        f"{hidden_dim}_" \
                        f"{mlp_dp_rate}_" \
                        f"{loss_fun}_" \
                        f"{lr}_" \
                        f"{weight_decay}"

    ckpt_dir = ckpt_root_dir + '/' + experiment_name
    os.makedirs(ckpt_dir, exist_ok=True)
    run_json_file = ckpt_dir + '/learning_curve.json'

    train_loss, valid_loss, val_target_mean_mse, val_target_median_mse = [], [], [], []
    val_target_mean_ranking_loss, val_target_median_ranking_loss = [], []

    print(experiment_name)
    
    if not os.path.exists(run_json_file):

        train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=True)
        
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=collate,
                                shuffle=False)

        # initialise the wandb logger and name your wandb project
        wandb.finish()

        wandb_logger = WandbLogger(project=projectname, save_dir=workdir)

        # add your batch size to the wandb config
        wandb_logger.experiment.config["random_seed"] = random_seed
        wandb_logger.experiment.config["batch_size"] = batch_size
        wandb_logger.experiment.config["node_input_dim"] = node_input_dim
        wandb_logger.experiment.config["edge_input_dim"] = edge_input_dim
        wandb_logger.experiment.config["num_heads"] = num_heads
        wandb_logger.experiment.config["num_layer"] = num_layer
        wandb_logger.experiment.config["dp_rate"] = dp_rate
        wandb_logger.experiment.config["layer_norm"] = layer_norm
        wandb_logger.experiment.config["batch_norm"] = batch_norm
        wandb_logger.experiment.config["residual"] = residual
        wandb_logger.experiment.config["hidden_dim"] = hidden_dim
        wandb_logger.experiment.config["mlp_dp_rate"] = mlp_dp_rate
        wandb_logger.experiment.config["loss_fun"] = loss_fun
        wandb_logger.experiment.config["lr"] = lr
        wandb_logger.experiment.config["weight_decay"] = weight_decay
        
        loss_function = None
        if loss_fun == 'mse':
            loss_function = torchmetrics.MeanSquaredError()
        elif loss_fun == 'binary':
            loss_function = torch.nn.BCELoss()

        if loss_function is None:
            return 999

        model_dict = {}
        with open(ckpt_dir + '/config.json', 'w') as fw:
            json.dump(dict(wandb_logger.experiment.config), fw, indent = 4)

        model = Gate(node_input_dim=node_input_dim,
                    edge_input_dim=edge_input_dim,
                    num_heads=num_heads,
                    num_layer=num_layer,
                    dp_rate=dp_rate,
                    layer_norm=layer_norm,
                    batch_norm=not layer_norm,
                    residual=True,
                    hidden_dim=hidden_dim,
                    mlp_dp_rate=mlp_dp_rate,
                    check_pt_dir=ckpt_dir,
                    batch_size=batch_size,
                    loss_function=loss_function,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    train_targets=targets_train_in_fold,
                    valid_targets=targets_val_in_fold,
                    subgraph_columns_dict=subgraph_columns_dict,
                    native_dfs_dict=native_dfs_dict,
                    log_train_mse=log_train_mse,
                    log_val_mse=log_val_mse)

        trainer = L.Trainer(accelerator='gpu',max_epochs=200, logger=wandb_logger, deterministic=True)

        # wandb_logger.watch(model)

        trainer.fit(model, train_loader, val_loader)

        valid_loss = model.learning_curve['valid_loss']
        train_loss = model.learning_curve['train_loss_epoch']
        val_target_mean_mse = model.learning_curve['val_target_mean_mse']
        val_target_median_mse = model.learning_curve['val_target_median_mse']
        val_target_mean_ranking_loss = model.learning_curve['val_target_mean_ranking_loss']
        val_target_median_ranking_loss = model.learning_curve['val_target_median_ranking_loss']

        with open(run_json_file, 'w') as f:
            f.write(json.dumps({'train_loss': train_loss, 
                                'valid_loss': valid_loss,
                                'val_target_mean_mse': val_target_mean_mse,
                                'val_target_median_mse': val_target_median_mse,
                                'val_target_mean_ranking_loss': val_target_mean_ranking_loss,
                                'val_target_median_ranking_loss': val_target_median_ranking_loss}, indent=4))

    else:
        with open(run_json_file) as f:
            data = json.load(f)
            train_loss = np.array(data['train_loss'])
            valid_loss = np.array(data['valid_loss'])
            val_target_mean_mse = np.array(data['val_target_mean_mse'])
            val_target_median_mse = np.array(data['val_target_median_mse'])
            val_target_mean_ranking_loss = np.array(data['val_target_mean_ranking_loss'])
            val_target_median_ranking_loss = np.array(data['val_target_median_ranking_loss'])

    loss1 = abs(train_loss[np.argmin(valid_loss)-1] - np.min(valid_loss))
    loss2 = np.min(valid_loss)
    penalty = 0
    if len(valid_loss) < 20:
        penalty += 1

    #loss3 = min(val_target_mean_mse[np.argmin(valid_loss)], val_target_median_mse[np.argmin(valid_loss)])
    #loss4 = min(val_target_mean_ranking_loss[np.argmin(valid_loss)], val_target_median_ranking_loss[np.argmin(valid_loss)])

    return loss1 + loss2 + penalty #+ loss3 + loss4

def cli_main():

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--scoredir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--dbdir', type=str, required=True)
    parser.add_argument('--labeldir', type=str, required=True)
    parser.add_argument('--log_train_mse', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--log_val_mse', default=False, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    args.gpus = 1

    dgldir = f"{args.outdir}/processed_data/dgl"
    labeldir = f"{args.outdir}/processed_data/label"
    folddir = f"{args.outdir}/fold{args.fold}"

    lines = open(folddir + '/targets.list').readlines()

    targets_train_in_fold = lines[0].split()
    targets_val_in_fold = lines[1].split()
    targets_test_in_fold = lines[2].split()

    print(f"Fold {args.fold}:")

    print(f"Train targets:")
    print(targets_train_in_fold)

    print(f"Validation targets:")
    print(targets_val_in_fold)

    print(f"Test targets:")
    print(targets_test_in_fold)

    start = time.time()
    train_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_train_in_fold)
    val_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_val_in_fold)
    # test_data = DGLData(dgl_folder=dgldir, label_folder=labeldir, targets=targets_test_in_fold)

    load_targets = []
    if args.log_train_mse:
        load_targets += targets_train_in_fold
    if args.log_val_mse:
        load_targets += targets_val_in_fold

    subgraph_columns_dict = read_subgraph_columns(args.datadir, load_targets)
    native_dfs_dict = read_native_dfs(args.labeldir, load_targets)
    end = time.time()
    
    print(f"Loading time: {end-start}s")

    random_seed = 3407

    L.seed_everything(random_seed, workers=True)

    projectname = f"{args.project}_fold{args.fold}"

    workdir = f"{args.dbdir}/{args.project}/fold{args.fold}"
    os.makedirs(workdir, exist_ok=True)

    ckpt_root_dir = workdir + '/ckpt/'
    os.makedirs(ckpt_root_dir, exist_ok=True)

    param_dict = {
        'random_seed': [random_seed],
        'projectname': [projectname],
        'workdir': [workdir],
        'train_data': [train_data],
        'val_data': [val_data],
        'num_heads': [4, 8],
        'num_layer': [1, 2, 3, 4, 5],
        'dp_rate': [0.2, 0.3, 0.4, 0.5],
        'hidden_dim': [16, 32, 64],
        'mlp_dp_rate': [0.2, 0.3, 0.4, 0.5],
        'loss_fun': ['mse'],
        'lr': [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
        'weight_decay': [0.01, 0.05],
        'layer_norm': [False, True],
        'batch_size': [256, 400, 512],
        'ckpt_root_dir': [ckpt_root_dir],
        'targets_train_in_fold': [targets_train_in_fold],
        'targets_val_in_fold': [targets_val_in_fold],
        'subgraph_columns_dict': [subgraph_columns_dict],
        'native_dfs_dict': [native_dfs_dict],
        'log_train_mse': [args.log_train_mse],
        'log_val_mse': [args.log_val_mse],
    }

    conf_Dict = dict()
    conf_Dict['batch_size'] = 1
    conf_Dict['num_iteration'] = 50
    conf_Dict['domain_size'] =20000
    conf_Dict['initial_random']= 5
    tuner = Tuner(param_dict, objfunc,conf_Dict)
    num_of_tries = 5
    all_runs = []

    for i in range(num_of_tries):
        results = tuner.minimize()
        print('best parameters:',results['best_params'])
        print('best objective:',results['best_objective'])

        all_runs.append(results)
        
        # saving the results
        save_res(all_runs,'mnist_mango.p')
    

if __name__ == '__main__':
    cli_main()
