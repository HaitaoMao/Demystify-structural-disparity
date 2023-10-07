import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.conv1 = GATConv(args.num_feat, args.hidden_dimension, args.num_of_heads, concat = True, dropout=.6)
        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        # self.conv2 = GATConv(args.hidden_dimension * args.num_of_heads, args.num_class, heads=1,
        #                       concat=False, dropout=0.6)
        self.layers = []
        self.bns = []
        if args.num_layers == 1:
            self.conv1 = GATConv(args.num_feat, args.hidden_dimension, args.num_of_heads, concat = False, dropout=args.attn_drop)
        else:
            self.conv1 = GATConv(args.num_feat, args.hidden_dimension, args.num_of_heads, concat = True, dropout=args.attn_drop)
            self.bns.append(torch.nn.BatchNorm1d(args.hidden_dimension * args.num_of_heads))
        self.layers.append(self.conv1)
        for _ in range(self.args.num_layers - 2):
            self.layers.append(
                GATConv(args.hidden_dimension * args.num_of_heads, args.hidden_dimension, args.num_of_heads, concat = True, dropout = args.dropout).cuda()
            )
            self.bns.append(torch.nn.BatchNorm1d(args.hidden_dimension * args.num_of_heads))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if args.num_layers > 1:
            self.layers.append(GATConv(args.hidden_dimension * args.num_of_heads, args.num_class, heads=args.num_of_out_heads,
                             concat=False, dropout=args.attn_drop).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.args.num_layers):
            x = F.dropout(x, self.args.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.args.num_layers - 1:
                if self.args.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x, None, None
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)
        # return x, None, None

    def set_others(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    def train_full_batch(self, data, epoch):
        self.train()
        self.optimizer.zero_grad()        
        out, _, _ = self.forward(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()