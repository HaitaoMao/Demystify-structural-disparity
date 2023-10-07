import torch 
from torch.nn import Linear
from torch_geometric.nn import APPNP
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, num_feat, hidden_dim, num_class, num_layers, dropout, with_bn, struct) -> None:
        super().__init__()
        self.layers = []
        self.bns = torch.nn.ModuleList()
        if num_layers == 1:
            self.layers.append(Linear(num_feat, num_class))
        else:
            self.layers.append(Linear(num_feat, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        if num_layers > 1:
            self.layers.append(Linear(hidden_dim, num_class))
        
        self.layers = torch.nn.ModuleList(self.layers)
        self.dropout = dropout
        self.with_bn = with_bn
        self.struct = struct
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        for l in self.bns:
            l.reset_parameters()

    def forward(self, x):
        ## special mlp for appnp
        if self.struct == 0:
            x = F.dropout(x, p = self.dropout, training=self.training)
            for i in range(len(self.layers)):
                if self.with_bn:
                    if i != len(self.layers) - 1:
                        x = F.relu(self.bns[i](self.layers[i](x)))
                    else:
                        if len(self.layers) > 1:
                            x = self.layers[i](F.dropout(x, p=self.dropout, training=self.training))
                        else:
                            x = self.layers[i](x)
                else:
                    if i != len(self.layers) - 1:
                        x = F.relu(self.layers[i](x))
                    else:
                        if len(self.layers) > 1:
                            x = self.layers[i](F.dropout(x, p=self.dropout, training=self.training))
                        else:
                            x = self.layers[i](x)
            return x
        else:
            for i in range(len(self.layers)):
                if i != len(self.layers) - 1:
                    if self.with_bn:
                        x = F.relu(self.bns[i](self.layers[i](x)))
                        x = F.dropout(x, p = self.dropout, training=self.training)
                    else:
                        x = F.relu(self.layers[i](x))
                        x = F.dropout(x, p=self.dropout, training=self.training)
                else:
                    x = self.layers[i](x)
            return x
        

class Net(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lin = MLP(args.num_feat, args.hidden_dimension, args.num_class, args.num_layers, args.dropout, args.with_bn, args.appnp_struct)
        self.prop1 = APPNP(args.K, args.alpha, dropout = args.appnp_drop)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1), None, None



    def set_others(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    def train_full_batch(self, data, epoch):
        self.train()
        # import ipdb;ipdb.set_trace()
        self.optimizer.zero_grad()        
        out, _, _ = self.forward(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()