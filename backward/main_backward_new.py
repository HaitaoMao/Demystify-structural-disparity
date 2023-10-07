import torch
from ogb.nodeproppred import Evaluator
from backward.algorithm.mlp import MLPnoreg
import logging
from copy import deepcopy
from torch_geometric.utils import subgraph






def main_backward(data, args):
    model = eval(args.algo_name)(args).cuda()
    model.set_others()
    best_val = 0
    best_test_acc = 0
    best_result = None
    best_y_pred = None
    model.eval()
    patience = args.patience
    accum = 0
    best_model = None
    if args.expmode == 'inductive':
        orig_data = deepcopy(data)
        train_edge_index, _ = subgraph(data.idx_obs, data.edge_index, relabel_nodes=True)
        data.edge_index = train_edge_index
        data.x = data.x[data.idx_obs]
        data.y = data.y[data.idx_obs]
        data.train_mask = data.obs_idx_train
        data.test_mask = data.obs_idx_test
        data.val_mask = data.obs_idx_val

    for epoch in range(1, 1 + args.epochs):
        torch.cuda.empty_cache()
        args.current_epoch = epoch
        
        #import ipdb; ipdb.set_trace()
        loss = model.train_full_batch(data, epoch)
        
        result = test(model, data, args=args)
        train_acc, valid_acc, test_acc, out, y_pred = result
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
        result = induc_test(best_model, orig_data, args)
        best_result, best_y_pred, ind_test_acc = result
    else:
        ind_test_acc = -1
    return best_test_acc, best_result, best_y_pred, ind_test_acc

@torch.no_grad()
def test(model, data, args=None):
    model.eval()
    # model.model.initialized = False
    out, _, _ = model(data)

    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
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