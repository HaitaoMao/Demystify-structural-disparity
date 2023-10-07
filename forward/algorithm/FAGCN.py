import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import FAConv
import numpy as np


class FAGCN(nn.Module):
    def __init__(self, args):
        super(FAGCN, self).__init__()
        self.args = args
        self.eps = args.eps
        self.layer_num = args.num_layers
        self.dropout = args.dropout
        self.hidden_num = args.hidden_dimension

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FAConv(self.hidden_num, self.eps, self.dropout))

        self.t1 = nn.Linear(args.num_feat, self.hidden_num)
        self.t2 = nn.Linear(self.hidden_num, args.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, raw, edge_index)
            # h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1), None, None

    def set_others(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    def train_full_batch(self, data, epoch):
        self.train()
        self.optimizer.zero_grad()        
        # import ipdb;ipdb.set_trace()
        out, _, _ = self.forward(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
