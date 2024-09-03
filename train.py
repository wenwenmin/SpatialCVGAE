import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

import torch
import random
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
import matplotlib.pyplot as plt

from VGAE_GCN.adj import graph
from VGAE_GCN.VGAE_model import VGAE
from VGAE_GCN.utils import *

def train_model(ann_data,
              input_dim,
              Conv_type = 'GCNConv',
              distType = 'Radius_balltree',
              rad_cutoff = 250,
              k_cutoff = 6,
              n_epochs = 1000,
              lr = 0.001,
              weight_decay = 0.0001,
              gradient_clipping=5.,
              verbose = True,
              seed = 0,
              mse_weight = 1,
              bce_weight = 0.1,
              kld_weight = 0.1,
              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
              ):

    seed_everthing(seed)

    ann_data.X = sp.csr_matrix(ann_data.X)

    net = graph(ann_data, distType=distType, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff)
    graph_dict = net.main()
    adj_norm = graph_dict['adj_norm'].to(device)
    adj_label = graph_dict['adj_label'].to(device)
    norm = graph_dict['norm_value']

    if 'highly_variable' in ann_data.var.columns:
        ann_data = ann_data[:, ann_data.var['highly_variable']]  # 将高可变基因提取出来赋给adata_Vars
    else:
        ann_data = ann_data

    if verbose:
        print('Size of input', ann_data.shape)
    if 'Spatial_Net' not in ann_data.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(ann_data)  # 将数据转化为pytorch可运算的形式

    model = VGAE(input_dim).to(device)

    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        h, conv, mu, logvar, z, rec_adj, rec_x, z_h = model(data.x, data.edge_index)
        mse_loss = F.mse_loss(rec_x,data.x)
        #loss = F.cross_entropy(de_feat ,data.adj_label) #GAE损失
        bce_loss = norm * F.binary_cross_entropy(rec_adj, adj_label)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / ann_data.shape[0] * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        loss = mse_weight * mse_loss + bce_weight * bce_loss + kld_weight * KLD
        loss_list.append(loss)
        loss.backward()
        # 梯度截断
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        #loss_list = torch.tensor([item.cpu().detach().numpy() for item in loss_list]).cuda()

    model.eval()
    h, conv, mu, logvar, z, rec_adj, rec_x, z_h = model(data.x, data.edge_index)

    z_rep = z.to('cpu').detach().numpy()
    ann_data.obsm['z'] = z_rep
    ReX = rec_x.to('cpu').detach().numpy()
    ReX[ReX < 0] = 0
    ann_data.obsm['ReX'] = ReX
    #z_x_rep = z_x.to('cpu').detach().numpy()
    #ann_data.obsm['z_x'] = z_x_rep

    return ann_data

def seed_everthing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


