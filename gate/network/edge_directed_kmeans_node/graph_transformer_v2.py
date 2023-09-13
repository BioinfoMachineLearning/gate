"""
@ Description: Evaluate DPROQ multitask
"""

import json
import dgl
from pathlib import Path
import torch
import torch.nn as nn
import torchmetrics
import lightning as L
from graph_transformer_edge_layer import GraphTransformerLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchmetrics
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class ResNetEmbedding(nn.Module):
    """Feature Learning Module"""
    def __init__(self, node_input_dim: int, edge_input_dim: int, out_dim: int):
        super(ResNetEmbedding, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        # net work module
        self.node_embedding = nn.Linear(node_input_dim, out_dim)
        self.bn_node = nn.BatchNorm1d(num_features=out_dim, eps=1e-08)
        self.edge_embedding = nn.Linear(edge_input_dim, out_dim)
        self.bn_edge = nn.BatchNorm1d(num_features=out_dim, eps=1e-08)
        self.relu = nn.LeakyReLU()

    def forward(self, node_feature, edge_feature):
        node_feature_embedded = self.node_embedding(node_feature)
        # print("node_feature_embedded")
        # if self.training:
        #     for i in range(16):
        #         print(node_feature_embedded[:,i])
        
        # print(self.bn_node(node_feature_embedded))
        node_feature_embedded = self.relu(self.bn_node(node_feature_embedded))
        # print("relu+bn")
        # print(node_feature_embedded)
        edge_feature_embedded = self.edge_embedding(edge_feature)
        edge_feature_embedded = self.relu(self.bn_edge(edge_feature_embedded))

        return node_feature_embedded, edge_feature_embedded


class MLP(nn.Module):
    """Read-out Module"""
    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5, L=2):
        super(MLP, self).__init__()
        self.L = L
        self.layers = nn.Sequential()
        for i in range(L):
            self.layers.add_module(f'Linear {i}', nn.Linear(input_dim // 2 ** i, input_dim // 2 ** (i + 1), bias=True))
            self.layers.add_module(f'BN {i}', nn.BatchNorm1d(input_dim // 2 ** (i + 1)))
            self.layers.add_module(f'relu {i}', nn.LeakyReLU())
            self.layers.add_module(f'dp {i}', nn.Dropout(p=dp_rate))
        self.final_layer = nn.Linear(input_dim // 2 ** L, output_dim, bias=True)

    def forward(self, x):
        x = self.layers(x)
        y = torch.sigmoid(self.final_layer(x))  # dockq_score
        return y


class Gate(L.LightningModule):
    """Gate model"""
    def __init__(self,
                 node_input_dim, 
                 edge_input_dim, 
                 num_heads, 
                 num_layer,
                 dp_rate, 
                 layer_norm,
                 batch_norm,
                 residual,
                 hidden_dim,
                 mlp_dp_rate,
                 check_pt_dir,
                 batch_size,
                 loss_function,
                 learning_rate,
                 weight_decay,
                 train_targets, valid_targets, datadir, labeldir):
        super().__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.num_heads = num_heads
        self.graph_n_layer = num_layer
        self.dp_rate = dp_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.hidden_dim = hidden_dim
        self.mlp_dp_rate = mlp_dp_rate
        self.check_pt_dir = check_pt_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.criterion = torchmetrics.MeanSquaredError()
        # self.criterion_node = torch.nn.BCELoss()
        # self.criterion_edge = torch.nn.BCELoss()
        self.criterion_node = loss_function

        self.resnet_embedding = ResNetEmbedding(self.node_input_dim,
                                                self.edge_input_dim,
                                                self.hidden_dim)
        
        self.graph_transformer_layer = nn.ModuleList(
            [GraphTransformerLayer(in_dim=self.hidden_dim,
                                   out_dim=self.hidden_dim,
                                   num_heads=self.num_heads,
                                   dropout=self.dp_rate,
                                   layer_norm=self.layer_norm,
                                   batch_norm=self.batch_norm,
                                   residual=self.residual,
                                   use_bias=True
                                   ) for _ in range(self.graph_n_layer)]
        )

        self.node_MLP_layer = MLP(input_dim=self.hidden_dim, output_dim=1, dp_rate=self.mlp_dp_rate)

        # self.edge_MLP_layer = MLP(input_dim=self.hidden_dim, output_dim=1, dp_rate=self.mlp_dp_rate)

        self.save_hyperparameters(ignore=['loss_function'])

        self.training_step_data_paths,  self.training_step_outputs = [], []
        self.valid_step_data_paths, self.valid_step_outputs = [], []
        self.train_targets = train_targets
        self.valid_targets = valid_targets
        self.datadir = datadir
        self.labeldir = labeldir

    def forward(self, g, node_feature, edge_feature):

        h, e = self.resnet_embedding(node_feature, edge_feature)

        # print(h)

        h = F.dropout(h, self.dp_rate, training=self.training)
        # e = F.dropout(e, self.dp_rate, training=self.training)

        for layer in self.graph_transformer_layer:
            h, e = layer(g, h, e)

        y1 = self.node_MLP_layer(h)

        # y2 = self.edge_MLP_layer(e)

        return y1

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        # optimizer = NoamOpt(self.hid_dim, 1, 2000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        scheduler = ReduceLROnPlateau(optimizer, mode='min')
        metric_to_track = 'valid_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }
    
    def configure_callbacks(self):
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor="valid_loss", dirpath=self.check_pt_dir, filename='{valid_loss:.5f}_{epoch}')
        early_stop = L.pytorch.callbacks.EarlyStopping(monitor="valid_loss", mode="min", patience=15)
        return [checkpoint_callback, early_stop]

    def save_graph_outputs(self, data_paths, pred_scores, train):
        if train:
            target_pred_subgraph_scores = self.train_target_pred_subgraph_scores
        else:
            target_pred_subgraph_scores = self.val_target_pred_subgraph_scores

        start_idx = 0
        for subgraph_path in data_paths:
            subgraph_filename = subgraph_path.split('/')[-1]
            targetname = subgraph_filename.split('_')[0]
            subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
                
            if targetname not in target_pred_subgraph_scores:
                target_pred_subgraph_scores[targetname] = {}

            subgraph_df = pd.read_csv(f"{self.datadir}/{targetname}/{subgraph_name.replace('.dgl', '.csv')}", index_col=[0])
            for i, modelname in enumerate(subgraph_df.columns):
                if modelname not in target_pred_subgraph_scores[targetname]:
                    target_pred_subgraph_scores[targetname][modelname] = []
                target_pred_subgraph_scores[targetname][modelname] += [pred_scores[start_idx + i]]
            start_idx += len(subgraph_df.columns)


    def training_step(self, batch, batch_idx):
        data, node_label, data_paths = batch
        # print(data_paths)
        # print(data.ndata['f'][0])
        node_out = self(data, data.ndata['f'], data.edata['f'])
        # print(node_out)

        node_loss = self.criterion_node(node_out, node_label)
        # print(node_loss)
        # edge_loss = self.criterion_edge(edge_out, edge_label)
        # loss = self.node_weight * node_loss + (1-self.node_weight) * edge_loss

        self.log('train_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        # self.log('train_node_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        # self.log('train_edge_loss', edge_loss, on_epoch=True, batch_size=self.batch_size)
        self.training_step_data_paths.append(data_paths)
        self.training_step_outputs.append(node_out)
        return node_loss
    
    def validation_step(self, batch, batch_idx):
        data, node_label, data_paths = batch
        # print(data_paths)
        # print(data.ndata['f'])
        node_out = self(data, data.ndata['f'], data.edata['f'])
        # print(node_out)

        node_loss = self.criterion_node(node_out, node_label)
        # edge_loss = self.criterion_edge(edge_out, edge_label)
        # loss = self.node_weight * node_loss + (1-self.node_weight) * edge_loss
        
        self.log('valid_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        # self.log('valid_node_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        # self.log('valid_edge_loss', edge_loss, on_epoch=True, batch_size=self.batch_size)
        self.valid_step_data_paths.append(data_paths)
        self.valid_step_outputs.append(node_out)
        return node_loss

    def on_train_epoch_end(self):
        target_pred_subgraph_scores = {}
        for subgraph_paths, pred_scores in zip(self.training_step_data_paths, self.training_step_outputs):
            pred_scores = pred_scores.cpu().data.numpy().squeeze(1)
            start_idx = 0
            for subgraph_path in subgraph_paths:
                subgraph_filename = subgraph_path.split('/')[-1]
                targetname = subgraph_filename.split('_')[0]
                subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
                    
                if targetname not in target_pred_subgraph_scores:
                    target_pred_subgraph_scores[targetname] = {}

                subgraph_df = pd.read_csv(f"{self.datadir}/{targetname}/{subgraph_name.replace('.dgl', '.csv')}", index_col=[0])
                for i, modelname in enumerate(subgraph_df.columns):
                    if modelname not in target_pred_subgraph_scores[targetname]:
                        target_pred_subgraph_scores[targetname][modelname] = []
                    target_pred_subgraph_scores[targetname][modelname] += [pred_scores[start_idx + i]]
                start_idx += len(subgraph_df.columns)

        self.training_step_outputs.clear()  # free memory
        self.training_step_data_paths.clear()  # free memory
        if len(self.train_targets) != len(target_pred_subgraph_scores):
            return

        target_mean_mse, target_median_mse = [], []
        for target in self.train_targets:
            ensemble_mean_scores, ensemble_median_scores = [], []
            for modelname in target_pred_subgraph_scores[target]:
                ensemble_mean_scores += [np.mean(np.array(target_pred_subgraph_scores[target][modelname].cpu()))]
                ensemble_median_scores += [np.median(np.array(target_pred_subgraph_scores[target][modelname].cpu()))]
            pred_df = pd.DataFrame({'model': list(target_pred_subgraph_scores[target].keys()), 'mean_score': ensemble_mean_scores, 
                                    'median_score': ensemble_median_scores})
            native_df = pd.read_csv(self.labeldir + '/' + target + '.csv')
            merge_df = pred_df.merge(native_df, on=f'model', how="inner")
            target_mean_mse += [mean_squared_error(np.array(merge_df['mean_score']), np.array(merge_df['tmscore']))]
            target_median_mse += [mean_squared_error(np.array(merge_df['median_score']), np.array(merge_df['tmscore']))]

        self.log('train_target_mean_mse', np.mean(np.array(target_mean_mse)), on_epoch=True)
        self.log('train_target_median_mse', np.mean(np.array(target_median_mse)), on_epoch=True)

    def on_validation_epoch_end(self):
        target_pred_subgraph_scores = {}
        for subgraph_paths, pred_scores in zip(self.valid_step_data_paths, self.valid_step_outputs):
            pred_scores = pred_scores.cpu().data.numpy().squeeze(1)
            start_idx = 0
            for subgraph_path in subgraph_paths:
                subgraph_filename = subgraph_path.split('/')[-1]
                targetname = subgraph_filename.split('_')[0]
                subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
                    
                if targetname not in target_pred_subgraph_scores:
                    target_pred_subgraph_scores[targetname] = {}

                subgraph_df = pd.read_csv(f"{self.datadir}/{targetname}/{subgraph_name.replace('.dgl', '.csv')}", index_col=[0])
                for i, modelname in enumerate(subgraph_df.columns):
                    if modelname not in target_pred_subgraph_scores[targetname]:
                        target_pred_subgraph_scores[targetname][modelname] = []
                    target_pred_subgraph_scores[targetname][modelname] += [pred_scores[start_idx + i]]
                start_idx += len(subgraph_df.columns)

        self.valid_step_outputs.clear()  # free memory
        self.valid_step_data_paths.clear()  # free memory
        if len(self.valid_targets) != len(target_pred_subgraph_scores):
            return

        target_mean_mse, target_median_mse = [], []
        for target in self.valid_targets:    
            ensemble_mean_scores, ensemble_median_scores = [], []
            for modelname in target_pred_subgraph_scores[target]:
                ensemble_mean_scores += [np.mean(np.array(target_pred_subgraph_scores[target][modelname]))]
                ensemble_median_scores += [np.median(np.array(target_pred_subgraph_scores[target][modelname]))]
            pred_df = pd.DataFrame({'model': list(target_pred_subgraph_scores[target].keys()), 'mean_score': ensemble_mean_scores, 
                                    'median_score': ensemble_median_scores})
            native_df = pd.read_csv(self.labeldir + '/' + target + '.csv')
            merge_df = pred_df.merge(native_df, on=f'model', how="inner")
            target_mean_mse += [mean_squared_error(np.array(merge_df['mean_score']), np.array(merge_df['tmscore']))]
            target_median_mse += [mean_squared_error(np.array(merge_df['median_score']), np.array(merge_df['tmscore']))]

        self.log('val_target_mean_mse', np.mean(np.array(target_mean_mse)), on_epoch=True)
        self.log('val_target_median_mse', np.mean(np.array(target_median_mse)), on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     data, target = batch
    #     out = self(data, data.ndata['f'], data.edata['f'])
    #     # print(out)
    #     # print(target)
    #     loss = self.criterion(out, target)
    #     self.log('test_loss', loss, on_epoch=True)
