import torch
from ogb.nodeproppred import Evaluator
from transfer.algorithm.student import student
import pickle
# import wandb
import ipdb
from helper.data_utils import read_npz
from copy import deepcopy
from torch_geometric.utils import subgraph

def main_transfer(data, args):
    model = eval(args.algo_name)(args).cuda()
    model_name_dict = {
            "APPNP": "PYGAPPNP-PYGAPPNP",
            "GAT": "PYGGAT-PYGGAT",
            "GCN": "PYGGCN-PYGGCN",
            "SGC": "PYGSGC-PYGSGC",
            "MLP_reg": "MLPReg-MLPReg",
            "MLP": "MLPnoreg-MLPnoreg"
    }
    # match the model and the performance
    if args.teacher not in model_name_dict.keys(): exit()
    model_name = model_name_dict
    teacher_model = model_name_dict[args.teacher]

    # num_split = 1 if args.dataset in ["arxiv", "product"] else 10
    if args.dataset in ["product", "arxiv", 'IGB_tiny']:
        num_split = 1
    elif args.dataset in ['genius', 'twitch-gamer']:
        num_split = 5
    else:
        num_split=10
    if args.teacher_res == "":
        if args.expmode == 'transductive':
            with open(f"./helper/best/{teacher_model}-{args.dataset}-{num_split}-1-res.pkl", "rb") as f:
                outs = pickle.load(f)
        else:
            with open(f"./helper/best/inductive/{teacher_model}-{args.dataset}-{num_split}-1-res.pkl", "rb") as f:
                outs = pickle.load(f)
    else:
        with open(args.teacher_res, 'rb') as f:
            outs = pickle.load(f)
    out = outs[args.split_seed] 
    out = torch.log_softmax(out, dim=-1)

    # ipdb.set_trace()
    train_teacher_acc = torch.sum(out[data.train_mask].argmax(dim=-1) == data.y[data.train_mask]) / torch.sum(data.train_mask)
    val_teacher_acc = torch.sum(out[data.val_mask].argmax(dim=-1) == data.y[data.val_mask]) / torch.sum(data.val_mask)
    test_teacher_acc = torch.sum(out[data.test_mask].argmax(dim=-1) == data.y[data.test_mask]) / torch.sum(data.test_mask)
    # import ipdb; ipdb.set_trace()
    print(f"Teacher: train:{train_teacher_acc}; val:{val_teacher_acc}; test: {test_teacher_acc}")
    if args.expmode == 'inductive':
        train_edge_index, _ = subgraph(data.idx_obs, data.edge_index)
        data.train_edge_index = train_edge_index
        orig_data = deepcopy(data)
        data.edge_index = data.train_edge_index
        data.train_mask = data.obs_idx_train
        data.test_mask = data.obs_idx_test
        data.val_mask = data.obs_idx_val
    model = eval(args.algo_name)(args).cuda()
    model.set_others()
    
    best_student_test , best_student_val = 0, 0
    out_student = None
    loss_p_list = []
    loss_t_list = []
    best_pred = None
    best_model = None
    for epoch in range(1, 1 + args.epochs):
        torch.cuda.empty_cache()
        args.current_epoch = epoch
        
        loss, loss_t, loss_p = model.train_student(data, out)
        out_test, result = model.test_student(data)

        train_acc, valid_acc, test_acc, pred = result
        if valid_acc > best_student_val:
            best_student_val = valid_acc
            best_student_test = test_acc
            out_student = out_test
            best_pred = pred
            best_model = model
    if args.expmode == 'inductive':
        result = induc_test(best_model, orig_data, args)
        _, _, _, _, _, ind_test_acc = result
    else:
        ind_test_acc = -1
    train_student_acc = torch.sum(torch.squeeze(best_pred)[data.train_mask] == data.y[data.train_mask]) / torch.sum(data.train_mask)
    val_student_acc = torch.sum(torch.squeeze(best_pred)[data.val_mask] == data.y[data.val_mask]) / torch.sum(data.val_mask)
    test_student_acc = torch.sum(torch.squeeze(best_pred)[data.test_mask] == data.y[data.test_mask]) / torch.sum(data.test_mask)
    print(f": train:{train_student_acc}; val:{val_student_acc}; test: {test_student_acc}")

    return test_teacher_acc.item(), best_student_test, out_student, best_pred, ind_test_acc
    

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
    train_acc = evaluator.eval({
        'y_true': y[data.obs_idx_train],
        'y_pred': y_pred[data.obs_idx_train],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[data.obs_idx_val],
        'y_pred': y_pred[data.obs_idx_val]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[data.obs_idx_test],
        'y_pred': y_pred[data.obs_idx_test],
    })['acc']
    ind_test_acc = evaluator.eval(
        {
        'y_true': y[data.idx_test_ind],
        'y_pred': y_pred[data.idx_test_ind]
        }
    )['acc']
    
    return train_acc, valid_acc, test_acc, out, y_pred, ind_test_acc


def test_homo(args, data, teacher_out, student_out, is_test):
    mask = data.test_mask if is_test else data.train_mask
    
    num_nodes = torch.sum(mask)
    teacher_out, student_out, labels = teacher_out[mask], student_out[mask], data.y[mask]

    teacher_predict = teacher_out.argmax(dim=-1)
    student_predict = student_out.argmax(dim=-1)
    teacher_correct = (teacher_predict == labels)
    student_correct = (student_predict == labels)

    tcsc = teacher_correct & student_correct
    twsc = ~teacher_correct & student_correct
    tcsw = teacher_correct & ~student_correct
    twsw = ~teacher_correct & ~student_correct


    tcsc_ratio = torch.sum(tcsc).item() / num_nodes
    twsc_ratio = torch.sum(twsc).item() / num_nodes
    tcsw_ratio = torch.sum(tcsw).item() / num_nodes
    twsw_ratio = torch.sum(twsw).item() / num_nodes

    name = "test" if is_test else "train"

    print(f"[{name}] tcsc: {tcsc_ratio}, twsc: {twsc_ratio}, tcsw: {tcsw_ratio}, twsw: {twsw_ratio},")
    student_advance = twsc # ~teacher_correct & student_correct
    teacher_advance = tcsw # teacher_correct & ~student_correct
    teacher_results, student_results = [], []
    for i, ratio_mask in enumerate(data.homo_masks):
        teacher_mask = ratio_mask[mask] & teacher_advance 
        teacher_ratio = torch.sum(teacher_mask) / torch.sum(ratio_mask)              # torch.sum(teacher_advance)
        teacher_results.append(teacher_ratio.item())
        student_mask = ratio_mask[mask] & student_advance 
        student_ratio = torch.sum(student_mask) / torch.sum(ratio_mask)             # torch.sum(student_advance)
        student_results.append(student_ratio.item())
    for homo_ratio, student_result, teacher_result in zip(data.homo_ratios, student_results, teacher_results):
        print(f"{homo_ratio}: teacher {teacher_result:4f}  student {student_result:4f}")
        # print(f"[{name}] student advance: {student_results:4f}")
        # print(f"[{name}] teacher advance: {teacher_results:4f}")


    return teacher_advance, student_advance