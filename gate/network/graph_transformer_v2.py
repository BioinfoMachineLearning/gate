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

class ResNetEmbedding(nn.Module):
    """Feature Learning Module"""
    def __init__(self, node_input_dim: int, edge_input_dim: int, out_dim: int):
        super(ResNetEmbedding, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        # net work module
        self.node_embedding = nn.Linear(node_input_dim, out_dim)
        self.bn_node = nn.BatchNorm1d(num_features=out_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, out_dim)
        self.bn_edge = nn.BatchNorm1d(num_features=out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, node_feature, edge_feature):
        node_feature_embedded = self.node_embedding(node_feature)
        node_feature_embedded = self.relu(self.bn_node(node_feature_embedded))
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
                 node_weight):
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
        self.node_weight = node_weight

        # self.criterion = torchmetrics.MeanSquaredError()
        self.criterion_node = torch.nn.BCELoss()
        self.criterion_edge = torch.nn.BCELoss()

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

        self.edge_MLP_layer = MLP(input_dim=self.hidden_dim, output_dim=1, dp_rate=self.mlp_dp_rate)

        self.save_hyperparameters()

    def forward(self, g, node_feature, edge_feature):

        h, e = self.resnet_embedding(node_feature, edge_feature)

        h = F.dropout(h, self.dp_rate, training=self.training)
        e = F.dropout(e, self.dp_rate, training=self.training)

        for layer in self.graph_transformer_layer:
            h, e = layer(g, h, e)

        y1 = self.node_MLP_layer(h)

        y2 = self.edge_MLP_layer(e)

        return y1, y2

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
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
        # checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor="valid_loss", dirpath=self.check_pt_dir, filename='{valid_loss:.5f}_{epoch}')
        early_stop = L.pytorch.callbacks.EarlyStopping(monitor="valid_loss", mode="min", patience=30)
        return [early_stop]

    def training_step(self, batch, batch_idx):
        data, node_label, edge_label = batch
        node_out, edge_out = self(data, data.ndata['f'], data.edata['f'])
        node_loss = self.criterion_node(node_out, node_label)
        edge_loss = self.criterion_edge(edge_out, edge_label)
        loss = self.node_weight * node_loss + (1-self.node_weight) * edge_loss

        self.log('train_loss', loss, on_epoch=True, batch_size=self.batch_size)
        self.log('train_node_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        self.log('train_edge_loss', edge_loss, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, node_label, edge_label = batch

        node_out, edge_out = self(data, data.ndata['f'], data.edata['f'])
        node_loss = self.criterion_node(node_out, node_label)
        edge_loss = self.criterion_edge(edge_out, edge_label)
        loss = self.node_weight * node_loss + (1-self.node_weight) * edge_loss
        
        self.log('valid_loss', loss, on_epoch=True, batch_size=self.batch_size)
        self.log('valid_node_loss', node_loss, on_epoch=True, batch_size=self.batch_size)
        self.log('valid_edge_loss', edge_loss, on_epoch=True, batch_size=self.batch_size)


    # def test_step(self, batch, batch_idx):
    #     data, target = batch
    #     out = self(data, data.ndata['f'], data.edata['f'])
    #     # print(out)
    #     # print(target)
    #     loss = self.criterion(out, target)
    #     self.log('test_loss', loss, on_epoch=True)
