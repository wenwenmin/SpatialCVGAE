import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch.nn import Dropout
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm, GCNConv, GATConv, SGConv, FeaStConv, SuperGATConv
from typing import Callable, Iterable, Union, Tuple, Optional
import logging


#DeepST类似
class VGAE(nn.Module):
    def __init__(self,
                 input_dim,  # 输入维度
                 Conv_type='GCNConv',
                 encoder_hidden=[1000,400,30],
                 conv_hidden = [64,8],
                 decoder_hidden=[400,1000],
                 p_drop=0.01,
                 ):
        super(VGAE, self).__init__()

        self.input_dim = input_dim
        self.encoder_hidden = encoder_hidden
        self. conv_hidden =  conv_hidden
        self.decoder_hidden = decoder_hidden
        self.p_drop = p_drop
        self.Conv_type = Conv_type

        self.conv1 = nn.Linear(input_dim, encoder_hidden[0])
        self.conv2 = nn.Linear(encoder_hidden[0], encoder_hidden[1])
        self.conv3 = nn.Linear(encoder_hidden[1], encoder_hidden[-1])
        #self.conv4 = nn.Linear(encoder_hidden[2], encoder_hidden[-1])


        self.act1 = nn.Sequential(BatchNorm(encoder_hidden[0]),
                                  nn.ReLU(),
                                  nn.Dropout(p_drop))
        self.act2 = nn.Sequential(BatchNorm(encoder_hidden[1]),
                                  nn.ReLU(),
                                  nn.Dropout(p_drop))
        self.act3 = nn.Sequential(BatchNorm(encoder_hidden[2]),
                                  nn.ReLU(),
                                  nn.Dropout(p_drop))

        self.act4 = nn.Sequential(BatchNorm(decoder_hidden[1]),
                                  nn.ReLU(),
                                  nn.Dropout(p_drop))

        #self.conv = Sequential('x, edge_index', [
        #    (GCNConv(encoder_hidden[-1], conv_hidden[0],), 'x, edge_index -> x1'),
        #])
        self.conv_x = Sequential('x, edge_index', [
            (GCNConv(encoder_hidden[-1], conv_hidden[0],), 'x, edge_index -> x1'),
        ])

        self.conv_mean = Sequential('x, edge_index', [
            (GCNConv(conv_hidden[0], conv_hidden[1],), 'x, edge_index -> x1'),
        ])
        self.conv_logvar = Sequential('x, edge_index', [
            (GCNConv(conv_hidden[0], conv_hidden[1],), 'x, edge_index -> x1'),
        ])

        self.conv_x_ = Sequential('x, edge_index', [
            (GCNConv(conv_hidden[1], decoder_hidden[0],), 'x, edge_index -> x1'),
        ])

        self.conv4 = nn.Linear(decoder_hidden[0], decoder_hidden[1])
        self.conv5 = nn.Linear(decoder_hidden[1], input_dim)



    def encode(self,
                x,
                adj
                ):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        h = self.act3(self.conv3(x)) #线性层最后一层
        conv = self.conv_x(h,adj)
        mu = self.conv_mean(conv,adj)
        logvar = self.conv_logvar(conv,adj)

        return h, conv, mu, logvar

    def reparameterize(self,
                        mu,
                        logvar,
                        ):
        std = torch.exp(logvar)  # exp（方差）
        eps = torch.randn_like(std)  # 随机构造和std同形的向量,并且符合标准正态分布
        z = eps.mul(std).add_(mu)
        return z

    def dot_product_decode(self,
                           z
                           ):
        rec_adj = torch.sigmoid(torch.matmul(z, z.t()))
        return rec_adj


    def forward(self,
                x,
                adj
                ):
        h, conv, mu, logvar = self.encode(x,adj)
        z = self.reparameterize(mu,logvar)
        z_h = torch.cat((h, z), 1)
        rec_adj = self.dot_product_decode(z)
        x = self.conv_x_(z, adj)
        x = self.act4(self.conv4(x))
        rec_x = self.conv5(x)


        return h, conv, mu, logvar, z, rec_adj, rec_x, z_h



