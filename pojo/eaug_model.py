# @time          : 2024/4/26 16:34
# @file          : eaug_model.py
# @Author        : wuqw
# @Description   :

import torch.nn as nn
import torch
from torch_geometric.nn import NNConv, GINEConv
import torch.nn.functional as F
from pojo.GATEConv import Edge_GATConv


class MPNN(nn.Module):

    def __init__(self, nf_input_dim, nf_hide_dim, nf_output_dim, layerNum, edge_dim):
        super(MPNN, self).__init__()
        self.layerNum = layerNum

        # 第一层
        edge_network1 = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ELU(),
            nn.Linear(edge_dim, nf_input_dim * nf_hide_dim)
        )
        self.convlist = nn.ModuleList()
        self.convlist.append(NNConv(nf_input_dim, nf_hide_dim, edge_network1, aggr="mean"))
        # 隐藏层
        for i in range(layerNum-1):
            edge_network = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ELU(),
                nn.Linear(edge_dim, nf_hide_dim * nf_hide_dim)
            )
            self.convlist.append(NNConv(nf_hide_dim, nf_hide_dim, edge_network, aggr="mean"))

        self.elu = nn.ELU()
        self.classifier = nn.Linear(nf_hide_dim, nf_output_dim)

    def forward(self, dglGraph, device):
        x = dglGraph.ndata['feature'].float().to(device)
        edge_attr = dglGraph.edata['feat'].float().to(device)
        srcindex, dstindex = dglGraph.edges()
        # 拼接这两个张量以得到edge_index格式
        edge_index = torch.stack([srcindex, dstindex], dim=0).to(device)
        # if ex.is_cuda:
        #     print("Tensor is on GPU")
        # else:
        #     print("Tensor is on CPU")
        for i in range(self.layerNum):
            x = self.convlist[i](x, edge_index, edge_attr)
            x = self.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)
        return x


class GIN(nn.Module):

    def __init__(self, nf_input_dim, nf_hide_dim, nf_output_dim, layerNum, edge_dim):
        super(GIN, self).__init__()
        self.layerNum = layerNum
        self.edge_nn_inputdim = nn.Linear(edge_dim, nf_input_dim)
        self.edge_nn_hidedim = nn.Linear(edge_dim, nf_hide_dim)
        # 第一层
        nn1 = nn.Sequential(
            nn.Linear(nf_input_dim, nf_hide_dim),
            nn.ELU(),
            nn.Linear(nf_hide_dim, nf_hide_dim*2),

        )
        self.convlist = nn.ModuleList()
        self.convlist.append(GINEConv(nn=nn1, edge_dim=edge_dim))
        # 隐藏层
        for i in range(layerNum-1):
            nn2 = nn.Sequential(
                nn.Linear(nf_hide_dim*2, nf_hide_dim * 2),
                nn.ELU(),
                nn.Linear(nf_hide_dim*2, nf_hide_dim * 2),
            )
            self.convlist.append(GINEConv(nn=nn2, edge_dim=edge_dim))
        self.elu = nn.ELU()
        self.classifier = nn.Linear(nf_hide_dim*2, nf_output_dim)

    def forward(self, dglGraph, device):

        x = dglGraph.ndata['feature'].float().to(device)
        edge_attr = dglGraph.edata['feat'].float().to(device)
        srcindex, dstindex = dglGraph.edges()
        edge_index = torch.stack([srcindex, dstindex], dim=0).to(device)
        for i in range(self.layerNum):
            x = self.convlist[i](x, edge_index, edge_attr)
            x = self.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)
        return x


class GAT(nn.Module):

    def __init__(self, nf_input_dim, nf_hide_dim, nf_output_dim, layerNum, edge_dim):
        super(GAT, self).__init__()
        self.layerNum = layerNum

        # 第一层
        self.convlist = nn.ModuleList()
        self.convlist.append(Edge_GATConv(nf_input_dim, nf_hide_dim, edge_dim))
        # 隐藏层
        for i in range(layerNum-1):
            self.convlist.append(Edge_GATConv(nf_hide_dim, nf_hide_dim, edge_dim))
        self.elu = nn.ELU()
        self.classifier = nn.Linear(nf_hide_dim, nf_output_dim)

    def forward(self, dglGraph, device):
        x = dglGraph.ndata['feature'].float().to(device)
        edge_attr = dglGraph.edata['feat'].float().to(device)
        srcindex, dstindex = dglGraph.edges()
        edge_index = torch.stack([srcindex, dstindex], dim=0).to(device)
        for i in range(self.layerNum):
            x = self.convlist[i](x, edge_index, edge_attr)
            x = self.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)
        return x
