from forward.model.sgc import SGC, sgc_precompute
from backward.model.MLPdrop import MLP
import torch
import torch.nn.functional as F
from torch_geometric.typing import Tensor, Optional, Tuple, Union
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import SGConv


def to_torch_coo_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
    :rtype: :class:`torch.sparse.FloatTensor`
    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_coo_tensor(edge_index)
        tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]]),
            values=tensor([1., 1., 1., 1., 1., 1.]),
            size=(4, 4), nnz=6, layout=torch.sparse_coo)
    """
    if size is None:
        size = int(edge_index.max()) + 1
    if not isinstance(size, (tuple, list)):
        size = (size, size)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    size = tuple(size) + edge_attr.size()[1:]
    out = torch.sparse_coo_tensor(edge_index, edge_attr, size,
                                  device=edge_index.device)
    out = out.coalesce()
    return out



class PYGSGC(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.feature_layer = args.K
        # self.sgc_layer = SGC(args.num_feat, args.num_class)
        if args.num_layers >= 1:
            hidden_dimensions = [args.hidden_dimension] * (args.num_layers - 1)
            
            self.hidden_dimensions = [args.hidden_dimension] + hidden_dimensions + [args.num_class]
            self.sgc_layer = MLP(args, self.hidden_dimensions)
        if args.num_layers >= 1:
            self.sgc = SGConv(args.num_feat, args.hidden_dimension, self.feature_layer)
        else:
            self.sgc = SGConv(args.num_feat, args.num_class, self.feature_layer)

    def set_others(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    def forward(self, data):
        ## support pyg data type
        features = data.x
        # edge_index, edge_weight = gcn_norm(  # yapf: disable
        #             data.edge_index, data.edge_weight, None, False,
        #             True, dtype=data.x.dtype)
        # sparse_adj = to_torch_coo_tensor(edge_index, edge_attr=edge_weight)
        # hidden = sgc_precompute(features, sparse_adj, self.feature_layer)
        hidden = self.sgc(features, data.edge_index)
        if self.args.num_layers >= 1:
            return self.sgc_layer(hidden), None, None
        else:
            return hidden, None, None
    
    def train_full_batch(self, data, epoch):
        self.optimizer.zero_grad()
        out, _, _ = self.forward(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()



class SGCNet(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.feature_layer = args.K
        # self.sgc_layer = SGC(args.num_feat, args.num_class)
        hidden_dimensions = [args.hidden_dimension] * (args.num_layers - 1)
        
        self.hidden_dimensions = [args.num_feat] + hidden_dimensions + [args.num_class]
        self.sgc_layer = MLP(args, self.hidden_dimensions)

    def set_others(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    def forward(self, data):
        ## support pyg data type
        features = data.x
        edge_index, edge_weight = gcn_norm(  # yapf: disable
                    data.edge_index, data.edge_weight, None, False,
                    True, dtype=data.x.dtype)
        sparse_adj = to_torch_coo_tensor(edge_index, edge_attr=edge_weight)
        hidden = sgc_precompute(features, sparse_adj, self.feature_layer)
        return self.sgc_layer(hidden), None, None
    
    def train_full_batch(self, data, epoch):
        self.optimizer.zero_grad()
        out, _, _ = self.forward(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()