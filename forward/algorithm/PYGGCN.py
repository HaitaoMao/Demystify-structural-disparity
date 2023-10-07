import torch 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        if args.num_layers == 1:
            self.convs.append(GCNConv(args.num_feat, args.num_class, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(args.num_feat, args.hidden_dimension, cached=False,
                             normalize=True))
            self.bns.append(torch.nn.BatchNorm1d(args.hidden_dimension))
            self.lns.append(torch.nn.LayerNorm(args.hidden_dimension, eps=1e-6))


            for _ in range(args.num_layers - 2):
                self.convs.append(GCNConv(args.hidden_dimension, args.hidden_dimension, cached=False,
                             normalize=True))
                self.lns.append(torch.nn.LayerNorm(args.hidden_dimension, eps=1e-6))
                self.bns.append(torch.nn.BatchNorm1d(args.hidden_dimension))

            self.convs.append(GCNConv(args.hidden_dimension, args.num_class, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.args.num_layers):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            if i != self.args.num_layers - 1:
                if self.args.with_ln:
                    x = self.lns[i](x)
                elif self.args.with_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
        return x, None, None 
    
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



