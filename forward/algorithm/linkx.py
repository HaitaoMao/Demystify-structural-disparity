import torch 
import torch.nn.functional as F
from torch_geometric.nn import LINKX as linkx

class LINKX(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # import ipdb;ipdb.set_trace()
        self.linkx = linkx(
            args.num_node, args.num_feat, args.hidden_dimension, args.num_class,
            args.num_layers, args.num_edge_layers, args.num_node_layers, args.dropout
        )


    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        #import ipdb;ipdb.set_trace()
        # self.linkx.num_nodes = x.shape[0]
        out = self.linkx(x, edge_index)
        return out, None, None 
    
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



