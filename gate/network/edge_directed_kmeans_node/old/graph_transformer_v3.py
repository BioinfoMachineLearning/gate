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
import time

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
        # self.final_layer1 = nn.Linear(input_dim // 2 ** L, output_dim, bias=True)
        self.final_layer2 = nn.Linear(input_dim // 2 ** L, output_dim, bias=True)

    def forward(self, x):
        x = self.layers(x)
        # y1 = torch.sigmoid(self.final_layer1(x))  # dockq_score
        y2 = self.final_layer2(x)
        return y2


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
                 loss_function_node_score,
                 loss_function_node_prob,
                 learning_rate,
                 weight_decay,
                 ce_loss_weight,
                 train_targets=[], valid_targets=[], 
                 subgraph_columns_dict={}, native_dfs_dict={},
                 log_train_mse=False, log_val_mse=False):
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
        self.criterion_node_score = loss_function_node_score
        self.criterion_node_prob = loss_function_node_prob
        self.training_step_data_paths,  self.training_step_outputs = [], []
        self.valid_step_data_paths, self.valid_step_outputs = [], []
        self.train_targets = train_targets
        self.valid_targets = valid_targets
        self.subgraph_columns_dict = subgraph_columns_dict
        self.native_dfs_dict = native_dfs_dict
        self.log_train_mse = log_train_mse
        self.log_val_mse = log_val_mse
        self.ce_loss_weight = ce_loss_weight

        self.learning_curve = {'train_loss_epoch': [], 'valid_loss_epoch': [], 
                               'val_target_mean_mse': [], 'val_target_median_mse': [],
                               'val_target_mean_ranking_loss': [], 'val_target_median_ranking_loss': []}

        self.save_hyperparameters(ignore=['loss_function_node_score', 'loss_function_node_prob', 'training_step_data_paths',
                                          'training_step_outputs', 'valid_step_data_paths',
                                          'valid_step_outputs', 'train_targets', 'valid_targets',
                                          'subgraph_columns_dict', 'native_dfs_dict', 'learning_curve'])

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
        return [checkpoint_callback] #, early_stop]

    def training_step(self, batch, batch_idx):
        data, node_label, data_paths, node_counts = batch
        # print(data_paths)
        # print(data.ndata['f'][0])
        node_prob = self(data, data.ndata['f'], data.edata['f'])
        # print(node_prob)
        # print(node_out)

        # node_score_loss = self.criterion_node_score(node_out, node_label)

        node_ce_loss = self.cal_ce_loss_for_batched_graphs(node_prob, node_label, node_counts)

        # node_loss = node_score_loss + self.ce_loss_weight * node_ce_loss

        # node_loss = node_ce_loss + node_score_loss / (node_score_loss/node_ce_loss).detach()

        node_loss = node_ce_loss

        self.log('train_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        # self.log('train_node_score_loss', node_score_loss, on_epoch=True, batch_size=self.batch_size)
        self.log('train_node_class_loss', node_ce_loss, on_epoch=True, batch_size=self.batch_size)

        if self.log_train_mse:
            self.training_step_data_paths.append(data_paths)
            self.training_step_outputs.append(node_out.cpu().data.numpy())
        
        return node_loss

    def cal_ce_loss_for_batched_graphs(self, output, true_scores, node_counts):

        ce_loss = 0.0
        start_node_idx = 0
        for node_count in node_counts:
            # print(start_node_idx)
            # print(node_count)
            node_prob = output[start_node_idx:start_node_idx+node_count]
            
            node_prob = torch.squeeze(node_prob, dim=1)
            print(node_prob)
            # print(node_prob.shape)
            node_class = torch.argmax(true_scores[start_node_idx:start_node_idx+node_count], dim=0)
            print(node_class[0])
            ce_loss_in_sub_graph = self.criterion_node_prob(node_prob, node_class[0])
            print(ce_loss_in_sub_graph)
            ce_loss += ce_loss_in_sub_graph
            start_node_idx += node_count

        return ce_loss / len(node_counts)

    def validation_step(self, batch, batch_idx):
        
        data, node_label, data_paths, node_counts = batch

        node_prob = self(data, data.ndata['f'], data.edata['f'])
        # print(node_prob)
        # node_score_loss = self.criterion_node_score(node_out, node_label)
        
        node_ce_loss = self.cal_ce_loss_for_batched_graphs(node_prob, node_label, node_counts)

        # node_loss = node_score_loss + self.ce_loss_weight * node_ce_loss

        # node_loss = node_ce_loss + node_score_loss / (node_score_loss/node_ce_loss).detach()

        node_loss = node_ce_loss

        self.log('valid_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        # self.log('valid_node_score_loss', node_score_loss, on_epoch=True, batch_size=self.batch_size)
        self.log('valid_node_class_loss', node_ce_loss, on_epoch=True, batch_size=self.batch_size)

        if self.log_val_mse:
            self.valid_step_data_paths.append(data_paths)
            self.valid_step_outputs.append(node_out.cpu().data.numpy())

        return node_loss
    
    def on_train_epoch_end(self):

        if not self.log_train_mse:
            return

        start = time.time()
        target_pred_subgraph_scores = {}
        for subgraph_paths, pred_scores in zip(self.training_step_data_paths, self.training_step_outputs):
            pred_scores = pred_scores.squeeze(1)
            start_idx = 0
            for subgraph_path in subgraph_paths:
                subgraph_filename = subgraph_path.split('/')[-1]
                targetname = subgraph_filename.split('_')[0]
                subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
                    
                if targetname not in target_pred_subgraph_scores:
                    target_pred_subgraph_scores[targetname] = {}

                subgraph_df_columns = self.subgraph_columns_dict[f"{targetname}_{subgraph_name.replace('.dgl', '')}"]
                for i, modelname in enumerate(subgraph_df_columns):
                    if modelname not in target_pred_subgraph_scores[targetname]:
                        target_pred_subgraph_scores[targetname][modelname] = []
                    target_pred_subgraph_scores[targetname][modelname] += [pred_scores[start_idx + i]]
                start_idx += len(subgraph_df_columns)

        self.training_step_outputs.clear()  # free memory
        self.training_step_data_paths.clear()  # free memory
        if len(self.train_targets) != len(target_pred_subgraph_scores):
            return
        
        target_mean_mse, target_median_mse = [], []
        for target in self.train_targets:
            ensemble_mean_scores, ensemble_median_scores = [], []
            for modelname in target_pred_subgraph_scores[target]:
                ensemble_mean_scores += [np.mean(np.array(target_pred_subgraph_scores[target][modelname]))]
                ensemble_median_scores += [np.median(np.array(target_pred_subgraph_scores[target][modelname]))]
            pred_df = pd.DataFrame({'model': list(target_pred_subgraph_scores[target].keys()), 'mean_score': ensemble_mean_scores, 
                                    'median_score': ensemble_median_scores})
            native_df = self.native_dfs_dict[target]
            merge_df = pred_df.merge(native_df, on=f'model', how="inner")
            target_mean_mse += [mean_squared_error(np.array(merge_df['mean_score']), np.array(merge_df['tmscore']))]
            target_median_mse += [mean_squared_error(np.array(merge_df['median_score']), np.array(merge_df['tmscore']))]

        end = time.time()
        # self.log('train_target_cal_time(s)', end - start, on_epoch=True)
        self.log('train_target_mean_mse', np.mean(np.array(target_mean_mse)), on_epoch=True)
        self.log('train_target_median_mse', np.mean(np.array(target_median_mse)), on_epoch=True)

    def on_validation_epoch_end(self):

        if not self.log_val_mse:
            return

        start = time.time()
        target_pred_subgraph_scores = {}
        for subgraph_paths, pred_scores in zip(self.valid_step_data_paths, self.valid_step_outputs):
            pred_scores = pred_scores.squeeze(1)
            start_idx = 0
            for subgraph_path in subgraph_paths:
                subgraph_filename = subgraph_path.split('/')[-1]
                targetname = subgraph_filename.split('_')[0]
                subgraph_name = subgraph_filename.split('_', maxsplit=1)[1]
                    
                if targetname not in target_pred_subgraph_scores:
                    target_pred_subgraph_scores[targetname] = {}

                subgraph_df_columns = self.subgraph_columns_dict[f"{targetname}_{subgraph_name.replace('.dgl', '')}"]
                for i, modelname in enumerate(subgraph_df_columns):
                    if modelname not in target_pred_subgraph_scores[targetname]:
                        target_pred_subgraph_scores[targetname][modelname] = []
                    target_pred_subgraph_scores[targetname][modelname] += [pred_scores[start_idx + i]]
                start_idx += len(subgraph_df_columns)

        self.valid_step_outputs.clear()  # free memory
        self.valid_step_data_paths.clear()  # free memory
        if len(self.valid_targets) != len(target_pred_subgraph_scores):
            return

        target_mean_mse, target_median_mse, target_mean_ranking_loss, target_median_ranking_loss = [], [], [], []
        for target in self.valid_targets:    
            ensemble_mean_scores, ensemble_median_scores = [], []
            for modelname in target_pred_subgraph_scores[target]:
                ensemble_mean_scores += [np.mean(np.array(target_pred_subgraph_scores[target][modelname]))]
                ensemble_median_scores += [np.median(np.array(target_pred_subgraph_scores[target][modelname]))]
            pred_df = pd.DataFrame({'model': list(target_pred_subgraph_scores[target].keys()), 'mean_score': ensemble_mean_scores, 
                                    'median_score': ensemble_median_scores})
            native_df = self.native_dfs_dict[target]
            native_tmscores = dict(zip(native_df['model'], native_df['tmscore']))

            merge_df = pred_df.merge(native_df, on=f'model', how="inner")
            target_mean_mse += [mean_squared_error(np.array(merge_df['mean_score']), np.array(merge_df['tmscore']))]
            
            pred_df = pred_df.sort_values(by=['mean_score'], ascending=False)
            pred_df.reset_index(inplace=True)
            top1_model = pred_df.loc[0, 'model']
            target_mean_ranking_loss += [float(np.max(np.array(native_df['tmscore']))) - float(native_tmscores[top1_model])]

            target_median_mse += [mean_squared_error(np.array(merge_df['median_score']), np.array(merge_df['tmscore']))]
            pred_df = pred_df.sort_values(by=['median_score'], ascending=False)
            pred_df.reset_index(inplace=True)
            top1_model = pred_df.loc[0, 'model']
            target_median_ranking_loss += [float(np.max(np.array(native_df['tmscore']))) - float(native_tmscores[top1_model])]
            

        end = time.time()
        # self.log('val_target_cal_time(s)', end - start, on_epoch=True)
        self.log('val_target_mean_mse', np.mean(np.array(target_mean_mse)), on_epoch=True)
        self.log('val_target_median_mse', np.mean(np.array(target_median_mse)), on_epoch=True)
        self.log('val_target_mean_ranking_loss', np.mean(np.array(target_mean_ranking_loss)), on_epoch=True)
        self.log('val_target_median_ranking_loss', np.mean(np.array(target_median_ranking_loss)), on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     data, target = batch
    #     out = self(data, data.ndata['f'], data.edata['f'])
    #     # print(out)
    #     # print(target)
    #     loss = self.criterion(out, target)
    #     self.log('test_loss', loss, on_epoch=True)
