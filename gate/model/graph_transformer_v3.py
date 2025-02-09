"""
@ Description: Evaluate DPROQ multitask
"""

import json
import dgl
from pathlib import Path
import torch
import torch.nn as nn
import lightning as L
from gate.model.graph_transformer_edge_layer import GraphTransformerLayer
import torch.nn.functional as F
import pandas as pd
import numpy as np
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
                 mlp_dp_rate):

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

    def forward(self, g, node_feature, edge_feature):

        h, e = self.resnet_embedding(node_feature, edge_feature)

        h = F.dropout(h, self.dp_rate, training=self.training)

        for layer in self.graph_transformer_layer:
            h, e = layer(g, h, e)

        y1 = self.node_MLP_layer(h)

        return y1