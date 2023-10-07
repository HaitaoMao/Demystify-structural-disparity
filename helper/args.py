import argparse
import torch
import numpy as np
import random
import sklearn
from helper.data_utils import get_dataset


def generate_args():
    parser = argparse.ArgumentParser(description='Understanding Graph Smoothness')

    parser.add_argument('--is_nni_tuning', type=int, default=0, help='whether use nni to tune the code, if 0, use parameter from params')
    parser.add_argument('--use_wandb_search', type=str, default="", help="path to the hyperparameter search file")
    
    parser.add_argument('--exp_name', type=str, default="sgc")
    parser.add_argument('--record_path', type=str, default="../record_hete")   # _hyper syn
    parser.add_argument('--data_path', type=str, default="../data")
    
    # model
    parser.add_argument('--train_schema', type=str, choices=["forward", "backward"], default="forward")
    parser.add_argument('--is_distill', type=int, default=0)

    parser.add_argument('--algo_name', type=str, default="SGCNet")
    parser.add_argument('--num_layers', type=int, default=1,
                            help='number of layers of the network')
    parser.add_argument('--hidden_dimension', type=int, default=64)
    parser.add_argument('--with_bn', type=int, default=0) #, action="store_true"
    parser.add_argument('--with_ln', type=int, default=0) #, action="store_true"
    parser.add_argument('--with_rn', type=int, default=0) # , action="store_true"
    # TODO: check here
    parser.add_argument('--act_fc', type=str, default='relu')
    parser.add_argument('--model_arch', nargs="*", type=int, required=False,
                      help='this term is for some specific model architecture design, do not neccery be true')
    # action='store',dest='list',

    # dataset 
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--num_fix', type=int, required=False)
    parser.add_argument('--ratio_fix', type=float, required=False)
    parser.add_argument('--is_fix', type=int, default=1)
    parser.add_argument('--num_split', type=int, default=1)
    parser.add_argument('--sweep_split', type=int, default=5)
    parser.add_argument('--is_new', type=int, default=1)

    
    # if not fixed
    # parser.add_argument('--val_ratio', type=float, default=0.2,help='ratio of validation set')
    parser.add_argument('--normalize_features', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.01,
                            help='input learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.00001) #0.0001
    parser.add_argument('--epochs', type=int, default=150,
                            help='input training epochs for training (default: 101)')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--sample_sizes', nargs="*", type=int, required=False,
                      help='this term is for some specific model architecture design, do not neccery be true')
    
    parser.add_argument('--random_seed', type=int, default=1,
                            help='input random seed for training (default: 1)')
    parser.add_argument('--dropout', type=float, default=0) #0.1
    parser.add_argument('--attn_drop', type=float, default=0)
    parser.add_argument('--appnp_struct', type=int, default=0)
    parser.add_argument('--appnp_drop', type=float, default=0)
        
    # common settings
    # parser.add_argument("--gpu_id", type=int, default=1)
    # model specific setting.0001)

    parser.add_argument("--lamda", type=float, default=0.05)
    parser.add_argument("--smoothidx", type=int, default=1)

    parser.add_argument("--alpha_type", type=str, default="APPNP" ) # APPNP, GCN, avg
    parser.add_argument("--num_prop_layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1) # APPNP, GCN, avg
        
    parser.add_argument("--num_hops", type=int, default=5)
    parser.add_argument("--step_len", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=2.0)

    parser.add_argument("--retain_grad", type=int, default=0)
    parser.add_argument("--is_hidden_record", type=int, default=0)
    parser.add_argument("--is_combine", type=int, default=0, help="how to record the GNN result")
    parser.add_argument("--is_analyze_norm", type=int, default=0, help="whether do the normalization on the feature when analyze.")
    parser.add_argument("--is_agg", type=int, default=0, help="whether do the normalization on the feature when analyze.")
    

    parser.add_argument("--n_points", type=int, default=1000)
    parser.add_argument("--n_features", type=int, default=50)
    parser.add_argument("--dist_sd_ratio", type=float, default=0.5)
    parser.add_argument("--sig_sq", type=float, default=0)   # the variance of the feature, default is set to zero
    parser.add_argument("--nonlinear", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--q", type=float, default=0.0005)
    parser.add_argument("--train_ratio", type=float, default=0.48)
    parser.add_argument("--val_ratio", type=float, default=0.32)
    parser.add_argument("--custom", type=int, default=0)

    parser.add_argument("--num_of_heads", type=int, default=8, help="num of heads parameters for GAT")
    parser.add_argument("--num_of_out_heads", type=int, default=1, help="num of out heads parameters for GAT")
    parser.add_argument("--bias", type=bool, default=True, help="bias for GAT")
    parser.add_argument("--log_attention_weights", action=argparse.BooleanOptionalAction, help="log att weights for GAT")
    parser.add_argument('--K', type=int, default=10, help='for appnp')
    parser.add_argument('--sweep_round', type=int, default=15, help='round of sweep')

    parser.add_argument('--best_run', action='store_true', default=False)
    parser.add_argument("--teacher", type=str, default="No")
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--best_path', type=str, default='./helper/best/')
    parser.add_argument('--save_path', type=str, default='./helper/test/')

    parser.add_argument('--generate_graph', action=argparse.BooleanOptionalAction)
    parser.add_argument('--generate_mode', type=str, default='sitao')
    parser.add_argument('--generation_config', default='./helper/generate_params.py')

    parser.add_argument('--is_synthetic', action=argparse.BooleanOptionalAction)
    parser.add_argument('--is_synthetic_full', type=int, default=0)
    parser.add_argument('--change', type=int, default=500)
    parser.add_argument('--class_ratio',type=float,default=0.05)
    parser.add_argument('--expmode', type=str, default='transductive')
    parser.add_argument('--inductive_rate', type=float, default = 0.7)
    parser.add_argument('--load_best_args', type=int, default=0)
    parser.add_argument('--runmode', type=str, default='single')
    parser.add_argument('--weight_decay2', type=float, default=5e-4)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--gcnii_type', type=int, default=0)
    parser.add_argument('--small_hete', type=int, default=0)
    parser.add_argument('--teacher_res', type=str, default="")
    parser.add_argument('--num_edge_layers', type=int, default=1)
    parser.add_argument('--num_node_layers', type=int, default=1)

    parser.add_argument('--is_extreme_mask', type=int, default=0)
    parser.add_argument('--syn_mask', type=str, default='test')
    parser.add_argument('--Init', type=str, default='PPR')
    parser.add_argument('--Gamma', type=float, default=None)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--is_ood', type=int, default=0)
    args = parser.parse_args()

    return args


def update_parameter(args, new_params):
    for key in new_params.keys():
        vars(args)[key] = new_params[key]

    return args


def dict_to_ns(dic):
    return argparse.Namespace(**dic)


def check_parameter(args):
    if args.dataset == "arxiv":
        args.with_bn = True
        args.with_res = True

    if args.is_fix:
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:   
            args.data_split = 1
        if args.dataset in ["Cornell", "Texas", "Wisconsin", "actor"]:
            args.num_data_split = 10
    
    return args


    
def generate_prefix(args):
    # return different prefix for generate the args
    # seed should be generate additional
    prefix_dict = {}

    prefix_dict["model"] = f"{args.algo_name}"
    # prefix_dict["model"] += f"{args.hidden_dimension}_{args.num_layers}"
    prefix_dict["train"] = f"{args.random_seed}"

    prefix_dict["dataset"] = f"{args.dataset}"
    
    prefix_dict["tuning"] = f"{args.lamda}_{args.sample_sizes}_{args.dropout}_{args.lr}"

    args.prefixs = prefix_dict
    
    return args


def set_seed_config(seed):
    # print('===========')
    # random seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
