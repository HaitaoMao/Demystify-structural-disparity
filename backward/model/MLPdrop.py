import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import ipdb

class MLP(nn.Module):
    def __init__(self, args, num_hiddens):
        super(MLP, self).__init__()
        self.args = args
        self.lins, self.bns = nn.ModuleList(), nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(len(num_hiddens) - 1):
            self.lins.append(nn.Linear(num_hiddens[i], num_hiddens[i+1]))
            if args.with_bn:
                self.bns.append(nn.BatchNorm1d(num_hiddens[i+1]))
            elif args.with_ln:
                self.lns.append(nn.LayerNorm(num_hiddens[i + 1], eps=1e-6))
        self.num_layer = len(self.lins)
        #ipdb.set_trace()
        self.intermediate_record = []
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.args.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
        elif self.args.with_ln:
            for ln in self.lns:
                ln.reset_parameters()
    
    def forward(self, hidden, is_retain_hidden=False):
        self.intermediate_record = []
        # if is_retain_hidden: self.intermediate_record.append(hidden)

        for i, lin in enumerate(self.lins):
            # hidden = F.dropout(hidden, p = self.args.dropout)
            hidden = lin(hidden)
            if i != (len(self.lins) - 1):
                if self.args.with_bn:
                    hidden = self.bns[i](hidden)  
                elif  self.args.with_ln:
                    hidden = self.lns[i](hidden)
                else:
                    hidden = hidden
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p = self.args.dropout, training=self.training)
            if is_retain_hidden:  self.intermediate_record.append(hidden)
        
        return hidden


