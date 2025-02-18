# 作者：wuqw
# 时间：2023/11/3 10:21

import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
from dgl.nn.pytorch import GINConv
import torch


class GCNModel(nn.Module):

    def __init__(self, inputNum, hideNum, outputNum, layerNum):
        super(GCNModel, self).__init__()
        self.layerNum = layerNum
        self.convlist = nn.ModuleList()
        self.convlist.append(GraphConv(inputNum, hideNum))
        for i in range(layerNum-1):
            self.convlist.append(GraphConv(hideNum, hideNum))
        self.classifier = nn.Linear(hideNum, outputNum)

    def forward(self, g):
        x = g.ndata['feature']
        # 卷积层数
        for i in range(self.layerNum):
            x = self.convlist[i](g, x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)
        return x


class GINNet(nn.Module):
    def __init__(self, inputNum, hideNum, outputNum, layerNum, dropout=0.7):
        super(GINNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layerNum = layerNum
        self.dropout = dropout

        # 使用更复杂的多层感知机作为映射函数
        mlp = Sequential(
            Linear(inputNum, hideNum),
            ReLU(),
            Linear(hideNum, hideNum * 2),
            ReLU(),
            Linear(hideNum * 2, hideNum * 2)
        )
        self.layers.append(GINConv(mlp, 'mean'))

        for i in range(layerNum - 1):
            mlp = Sequential(
                Linear(hideNum * 2, hideNum * 2),
                ReLU(),
                Linear(hideNum * 2, hideNum * 2),
                ReLU(),
                Linear(hideNum * 2, hideNum * 2)
            )
            self.layers.append(GINConv(mlp, 'mean'))

        self.linear = nn.Linear(hideNum * 2, outputNum)
        self.dropout_layer = Dropout(dropout)

    def forward(self, g):
        h = g.ndata['feature'].float()
        for i in range(self.layerNum):
            h = self.layers[i](g, h)
            h = F.relu(h)
            h = self.dropout_layer(h)
        h = self.linear(h)
        return h


class A_GCN(torch.nn.Module):
    def __init__(self, in_num, hid_num, N_layer):
        super(A_GCN, self).__init__()

        self.N_layer = N_layer
        self.GCL_List = nn.ModuleList()
        self.GCL_List.append(GraphConv(in_num, hid_num))
        for i in range(N_layer - 1):
            self.GCL_List.append(GraphConv(hid_num, hid_num))

    def forward(self, g):
        x = g.ndata['feature']
        X_List = []
        for i in range(self.N_layer):
            x = self.GCL_List[i](g, x)
            X_List.append(x)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)

        return x


class BiGCN_N(torch.nn.Module):
    def __init__(self, in_num, hid_num, out_num, N_layer):
        self.N_layer = N_layer
        super(BiGCN_N, self).__init__()
        self.FW_GCN = A_GCN(in_num, hid_num, N_layer)
        self.BW_GCN = A_GCN(in_num, hid_num, N_layer)
        self.fc1 = nn.Linear(hid_num * 2, out_num)

    def forward(self, fw_g, bw_g):
        fx = self.FW_GCN(fw_g)
        bx = self.BW_GCN(bw_g)

        x = torch.cat((fx, bx), 1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)

        return x
