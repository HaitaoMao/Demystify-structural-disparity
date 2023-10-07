import torch
from ogb.nodeproppred import Evaluator
from forward.algorithm.SGC import SGCNet, PYGSGC
from forward.algorithm.PYGGCN import GCN as PYGGCN
from forward.algorithm.PYGGAT import GAT as PYGGAT
from forward.algorithm.APPNP import Net as PYGAPPNP
from forward.algorithm.H2GCN import H2GCN 
from forward.algorithm.FAGCN import FAGCN
from forward.algorithm.GCNII import GCNII
from forward.algorithm.linkx import LINKX
from forward.algorithm.GPRGNN import GPRGNN
from forward.algorithm.GPRGNN2 import GPRGNN2
from torch_geometric.utils import subgraph
from copy import deepcopy
from helper.data_utils import extre_mask



def idx_to_mask(idx):
    mask = torch.zeros(idx.shape[0], dtype=torch.bool)
    mask[idx] = True
    return mask


def main_forward(data, args):
    if args.expmode == 'inductive':
        orig_data = deepcopy(data)
        train_edge_index, _ = subgraph(data.idx_obs, data.edge_index, relabel_nodes=True)
        data.edge_index = train_edge_index
        data.x = data.x[data.idx_obs]
        data.y = data.y[data.idx_obs]
        data.train_mask = data.obs_idx_train
        data.test_mask = data.obs_idx_test
        data.val_mask = data.obs_idx_val
    try:
        if args.is_extreme_mask:
            data = extre_mask(args, None)
    except Exception:
        pass
    vars(args)["num_node"] = data.x.shape[0]
    vars(args)["num_feat"] = data.x.shape[1]
    vars(args)["num_class"] = max(data.y).item() + 1
    model = eval(args.algo_name)(args).cuda()
    model.set_others()
    best_val = 0
    best_test_acc = 0
    best_result = None
    best_y_pred = None
    best_ind_acc = 0
    model.eval()
    patience = args.patience
    accum = 0
    best_model = model
        # import ipdb; ipdb.set_trace()
    for epoch in range(1, 1 + args.epochs):
        torch.cuda.empty_cache()
        args.current_epoch = epoch
        
        #import ipdb; ipdb.set_trace()
        loss = model.train_full_batch(data, epoch)
        result = test(model, data, args=args)
        train_acc, valid_acc, test_acc, out, y_pred = result
        # print(loss, train_acc, valid_acc, test_acc)
        
        if valid_acc > best_val:
            accum = 0
            best_val = valid_acc
            best_test_acc = test_acc
            best_result = out
            best_y_pred = y_pred
            best_model = model
        else:
            accum += 1
            if accum > patience:
                break
    if args.expmode == 'inductive':
        # import ipdb;ipdb.set_trace()
        if args.algo_name == 'H2GCN':
            best_model.model.initialized = False
        result = induc_test(best_model, orig_data, args)
        best_result, best_y_pred, ind_test_acc = result
        data = orig_data
    else:
        ind_test_acc = -1
    return best_test_acc, best_result, best_y_pred, ind_test_acc, best_model

@torch.no_grad()
def test(model, data, args=None):
    model.eval()
    out, _, _ = model(data)

    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    # ipdb.set_trace()
    train_acc = evaluator.eval({
        'y_true': y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[data.val_mask],
        'y_pred': y_pred[data.val_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']
    
    return train_acc, valid_acc, test_acc, out, y_pred



@torch.no_grad()
def induc_test(model, data, args=None):
    model.eval()
    out, _, _ = model(data)

    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    ind_test_acc = evaluator.eval(
        {
        'y_true': y[data.idx_test_ind],
        'y_pred': y_pred[data.idx_test_ind]
        }
    )['acc']
    
    return out, y_pred, ind_test_acc










