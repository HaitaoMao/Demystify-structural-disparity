import sys
sys.path.append('..')
import numpy as np
from collections import defaultdict
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid
from helper.data_utils import get_dataset, save_best
from helper.args import generate_args
from torch_geometric.utils import homophily, k_hop_subgraph, to_dense_adj, remove_self_loops, to_networkx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
import ipdb 
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os.path as osp
from torch_scatter import scatter_add
import torch_sparse


def compute_homo_mask_new(data, args, is_hete):
    # import ipdb; ipdb.set_trace()
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    edge_homo_value = (data.y[row] == data.y[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg

    mask_begin = (results >= 0)
    
    ends = [0.2, 0.4, 0.6, 0.8, 1.0]
    if is_hete:
        ends = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    masks = []
    for end in ends:
        mask_end = (results <= end)

        mask = mask_begin * mask_end
        masks.append(mask)
        mask_begin = (results >= end)
    
    return results, masks, ends


def category_node_by_label(data):
    # ipdb.set_trace()
    labels = data.y.tolist()
    dict_with_idx = defaultdict(list)
    for pos, ele in enumerate(labels):
        dict_with_idx[ele].append(pos)
    return dict_with_idx


def category_node_by_label_part(data, ratio, mask, mode):
    # ipdb.set_trace()
    label = data.y.numpy()
    # sortorig_label = label
    dict_with_idx = defaultdict(list)
    node_homo_ratio, _, _ = compute_homo_mask_new(data, None, False)
    node_homo_ratio = node_homo_ratio.numpy()
    if mode == 'tohete':
        sort_idx = np.argsort(node_homo_ratio)[::-1]
    else:
        sort_idx = np.argsort(node_homo_ratio)
    labels = label[sort_idx].tolist()
    selected_nodes = len(torch.tensor(labels)[mask])
    num_per_class = int(selected_nodes / (max(labels) + 1) * ratio)
    total_needed = (max(labels) + 1) * num_per_class
    add = 0
    for pos, ele in enumerate(labels):
        if len(dict_with_idx[ele]) < num_per_class:
            if mask[sort_idx[pos]]:
                dict_with_idx[ele].append(sort_idx[pos])
                add += 1
                if add == total_needed:
                    break
        else:
            continue
    return dict_with_idx

## we first try sitao's graph generation


def generate_base_features(data, num_node_total, pattern='random', num_each_class = 400, total_class = 5):
    if pattern == "random":
        return torch.from_numpy(np.random.rand(num_node_total, 1433)).float()
    else:
        features, labels = data.x, data.y
        nclass = labels.max().item() + 1
        column_idx = [np.where(labels == i % nclass)[0] for i in range(5)]
        idx = []
        for j in range(total_class):
            if column_idx[j].shape[0] > num_each_class:
                idx = (
                    idx
                    + np.random.choice(column_idx[j], num_each_class, replace=False).tolist()
                )
            else:
                idx = (
                    idx
                    + column_idx[j].tolist()
                    + np.random.choice(
                        column_idx[j], num_each_class - column_idx[j].shape[0], replace=False
                    ).tolist()
                )
        return features[np.array(idx), :]


def generate_output_label(num_class, node_per_class):
    label = np.eye(num_class)
    return np.repeat(label, repeats=node_per_class, axis=0)



def generate_graph(
    num_class, num_node_total, degree_intra, num_graph, graph_type, edge_homos, base_dir = '/mnt/home/haitaoma/Graph-smooth/code/synthetic_data/sitao'
):
    node_per_class = num_node_total // num_class
    base_data_dir = f"{base_dir}/{graph_type}"
    Path(base_data_dir).mkdir(parents=True, exist_ok=True)

    if graph_type == "regular":
        for edge_homo in edge_homos:
            for graph_num in range(num_graph):
                # logger.log_init(f"Generating regular graph {graph_num} with edge homophily: {edge_homo}")
                degree_inter = int(degree_intra / edge_homo - degree_intra)
                output_label = generate_output_label(num_class, node_per_class)
                adj_matrix = np.zeros((num_node_total, num_node_total))
                for i in range(num_class):
                    for j in range(i * node_per_class, (i + 1) * node_per_class):
                        # generate inner class adjacency
                        adj_matrix[
                            j,
                            random.sample(
                                set(
                                    np.delete(
                                        np.arange(i * node_per_class, (i + 1) * node_per_class),
                                        np.where(
                                            np.arange(i * node_per_class, (i + 1) * node_per_class)
                                            == j
                                        ),
                                    )
                                ),
                                degree_intra,
                            ),
                        ] = 1
                        # generate cross class adjacency
                        adj_matrix[
                            j,
                            random.sample(
                                set(
                                    np.delete(
                                        np.arange(0, num_node_total),
                                        np.arange(i * node_per_class, (i + 1) * node_per_class),
                                    )
                                ),
                                degree_inter,
                            ),
                        ] = 1

                degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

                # save generated graph matrices
                save_graphs(base_data_dir, edge_homo, graph_num, adj_matrix, degree_matrix, output_label)

    else:
        for edge_homo in edge_homos:
            for graph_num in range(num_graph):
                # logger.log_init(f"Generating regular graph {graph_num} with edge homophily: {edge_homo}")
                degree_matrix = np.zeros((num_node_total, num_node_total))
                output_label = generate_output_label(num_class, node_per_class)
                adj_matrix = np.zeros((num_node_total, num_node_total))
                for i in range(num_class):
                    # generate inner class adjacency
                    num_edge_same = degree_intra * node_per_class
                    adj_in_class = np.zeros((node_per_class, node_per_class))
                    adj_up_elements = np.array(
                        [1] * (int(num_edge_same / 2))
                        + [0]
                        * (
                            int(
                                (node_per_class - 1) * node_per_class / 2
                                - num_edge_same / 2
                            )
                        )
                    )
                    np.random.shuffle(adj_up_elements)
                    adj_in_class[np.triu_indices(node_per_class, 1)] = adj_up_elements
                    adj_in_class = adj_in_class + adj_in_class.T
                    adj_matrix[
                        node_per_class * i: node_per_class * (i + 1),
                        node_per_class * i: node_per_class * (i + 1),
                    ] = adj_in_class

                    # generate cross class adjacency
                    if i != num_class - 1:
                        if i == 0:
                            node_out_class = (
                                round(num_edge_same * (1 - edge_homo) / edge_homo) + 1
                            )
                        else:
                            existing_out_class_edges = np.sum(
                                adj_matrix[
                                    node_per_class * i : node_per_class * (i + 1),
                                    0: node_per_class * (i),
                                ]
                            )
                            node_out_class = (
                                round(
                                    num_edge_same * (1 - edge_homo) / edge_homo
                                    - existing_out_class_edges
                                )
                                + 1
                            )
                        adj_out_elements = np.array(
                            [1] * (node_out_class)
                            + [0]
                            * (
                                (num_class - 1 - i) * node_per_class ** 2
                                - node_out_class
                            )
                        )
                        np.random.shuffle(adj_out_elements)
                        adj_out_elements = adj_out_elements.reshape(
                            node_per_class, (num_class - 1 - i) * node_per_class
                        )
                        adj_matrix[
                            node_per_class * i : node_per_class * (i + 1),
                            node_per_class * (i + 1): node_per_class * (num_class),
                        ] = adj_out_elements
                        adj_matrix[
                            node_per_class * (i + 1): node_per_class * (num_class),
                            node_per_class * i: node_per_class * (i + 1),
                        ] = adj_out_elements.T
                    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

                # save generated graph matrices
                save_graphs(base_data_dir, edge_homo, graph_num, adj_matrix, degree_matrix, output_label)


def combine_generation(num_class, total_node, pattern = 'homo->hete'):
    node_per_class = total_node // num_class
    



def save_graphs(
    base_data_dir, edge_homo, graph_num, adj_matrix, degree_matrix, output_label
):
    adj_matrix = torch.tensor(adj_matrix).to_sparse()
    degree_matrix = torch.tensor(degree_matrix).to_sparse()
    output_label = torch.tensor(output_label).to_sparse()

    DATA_DIR = f"{base_data_dir}/{edge_homo:.3f}"
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    torch.save(
        adj_matrix,
        f"{DATA_DIR}/adj_{edge_homo}_{graph_num}.pt",
    )
    torch.save(
        degree_matrix,
        f"{DATA_DIR}/degree_{edge_homo}_{graph_num}.pt",
    )
    torch.save(output_label, f"{DATA_DIR}/label_{edge_homo}_{graph_num}.pt")



def local_hetero_edge_addition(data, ratio, edges, node_label_cat = None, mode='tohete', dataset = 'Cora', mask = 'train', with_noise = False, noise_level = 0.5):
    """
        Yao's paper Page 6, Alg.1 
    """
    ### predefined $D_c$ for each dataset(Cora, Citeseer, Squirrel, Chamelon)
    ### any d_c that's distinguishable enough can be used here
    if mask == 'test':
        if dataset == 'Cora' or dataset == 'PubMed':
            mask_selected = data.test_masks[0]
        else:
            mask_selected = data.test_mask[:, 0]
    else:
        if dataset == 'Cora' or dataset == 'PubMed':
            mask_selected = data.train_masks[0]
            mask_val = data.val_masks[0]
            mask_selected = mask_val | mask_selected
        else:
            mask_selected = data.train_mask[:, 0]
            mask_val = data.val_mask[:, 0]
            mask_selected = mask_val | mask_selected
    cora_d_c = np.array([[0,0.5,0,0,0,0,0.5],
                [0.5,0,0.5,0,0,0,0],
                [0,0.5,0,0.5,0,0,0],
                [0,0,0.5,0,0.5,0,0],
                [0,0,0,0.5,0,0.5,0],
                [0,0,0,0,0.5,0,0.5],
                [0.5,0,0,0,0,0.5,0]]) # type: ignore
    citeseer_d_c = np.array([[0,0.5,0,0,0,0.5],
                            [0.5,0,0.5,0,0,0],
                            [0,0.5,0,0.5,0,0],
                            [0,0,0.5,0,0.5,0],
                            [0,0,0,0.5,0,0.5],
                            [0.5,0,0,0,0.5,0]]) # type: ignore
    squi_chame_d_c = np.array([[0,0.5,0,0,0.5],
                            [0.5,0,0.5,0,0],
                            [0,0.5,0,0.5,0],
                            [0,0,0.5,0,0.5],
                            [0.5,0,0,0.5,0]]) # type: ignore


    edge_index = data.edge_index
    device = edge_index.device
    dataset_choice = {'Cora': cora_d_c, 'CiteSeer': citeseer_d_c, 'Squirrel': squi_chame_d_c, "Chameleon": squi_chame_d_c}
    d_c = dataset_choice.get(dataset, "Cora")
    total_mask_num = data.x[mask_selected].shape[0]
    total_class_num = data.y.max().item() + 1
    req_num = int(total_mask_num / total_class_num * ratio)
    if not node_label_cat:
        node_label_cat = category_node_by_label_part(data, ratio, mask_selected, mode)
    else:
        node_label_cat = node_label_cat
    row = []
    col = []
    total_add = 0
    cache = {}
    while total_add < edges:
        if with_noise:
            r = np.random.uniform(size = 1)[0]
        total_label_class = list(node_label_cat.keys())
        sampled_class = np.random.choice(total_label_class, size = 1, replace=False)[0]
        # node_idxs = np.arange(data.x.shape[0])
        sampled_node_idx = np.random.choice(node_label_cat[sampled_class], size=1, replace=False)[0]
        node_label = data.y[sampled_node_idx].item()
        d_yi = d_c[node_label]
        if mode == 'tohete':
            if with_noise and r <= noise_level:
                noise_label = list(set(data.y[data.y != node_label].tolist()))
                sampled_node_label = np.random.choice(noise_label, size = 1, replace=False)[0]
            else:
                sampled_node_label = np.random.choice(np.arange(d_yi.shape[0]), size = 1, replace=False, p=d_yi)[0] 
        else:
            sampled_node_label = node_label
        new_node_idx = np.random.choice(node_label_cat[sampled_node_label], size = 1, replace=False)[0]
        if not cache.get((min(sampled_node_idx, new_node_idx), max(sampled_node_idx, new_node_idx)), None):
            row.append(sampled_node_idx)
            col.append(new_node_idx)
            cache[(min(sampled_node_idx, new_node_idx), max(sampled_node_idx, new_node_idx))] = True
            total_add += 1
        

    row = torch.IntTensor(row)
    col = torch.IntTensor(col)
    edge_index_to_add = torch.stack([row, col], dim = 0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim = 1)
    edge_index = to_undirected(edge_index)
    data.edge_index = edge_index
    data.original_index = node_label_cat
    return data



def hetero_edge_addition(data, K, dataset = 'Cora', with_noise = False, noise_level = 0.5, mode='tohete'):
    """
        Yao's paper Page 6, Alg.1 
    """
    ### predefined $D_c$ for each dataset(Cora, Citeseer, Squirrel, Chamelon)
    ### any d_c that's distinguishable enough can be used here
    cora_d_c = np.array([[0,0.5,0,0,0,0,0.5],
                [0.5,0,0.5,0,0,0,0],
                [0,0.5,0,0.5,0,0,0],
                [0,0,0.5,0,0.5,0,0],
                [0,0,0,0.5,0,0.5,0],
                [0,0,0,0,0.5,0,0.5],
                [0.5,0,0,0,0,0.5,0]])
    citeseer_d_c = np.array([[0,0.5,0,0,0,0.5],
                            [0.5,0,0.5,0,0,0],
                            [0,0.5,0,0.5,0,0],
                            [0,0,0.5,0,0.5,0],
                            [0,0,0,0.5,0,0.5],
                            [0.5,0,0,0,0.5,0]])
    squi_chame_d_c = np.array([[0,0.5,0,0,0.5],
                            [0.5,0,0.5,0,0],
                            [0,0.5,0,0.5,0],
                            [0,0,0.5,0,0.5],
                            [0.5,0,0,0.5,0]])


    edge_index = data.edge_index
    device = edge_index.device
    dataset_choice = {'Cora': cora_d_c, 'CiteSeer': citeseer_d_c, 'Squirrel': squi_chame_d_c, "Chamelon": squi_chame_d_c}
    d_c = dataset_choice.get(dataset, "Cora")
    node_label_cat = category_node_by_label(data)
    row = []
    col = []
    cache = {}
    total_add = 0
    while total_add < K:
        if with_noise:
            r = np.random.uniform(size = 1)[0]
        node_idxs = np.arange(data.x.shape[0])
        sampled_node_idx = np.random.choice(node_idxs, size=1, replace=False)[0]
        node_label = data.y[sampled_node_idx]
        if mode == 'tohete':
            d_yi = d_c[node_label]
            if with_noise and r <= noise_level:
                noise_label = list(set(data.y[data.y != node_label].tolist()))
                sampled_node_label = np.random.choice(noise_label, size = 1, replace=False)[0]
            else:
                sampled_node_label = np.random.choice(np.arange(d_yi.shape[0]), size = 1, replace=False, p=d_yi)[0] 
            new_node_idx = np.random.choice(node_label_cat[sampled_node_label], size = 1, replace=False)[0]
        else:
            sampled_node_label = node_label.item()
            # ipdb.set_trace()
            new_node_idx = np.random.choice(node_label_cat[sampled_node_label], size = 1, replace=False)[0]
        if not cache.get((min(sampled_node_idx, new_node_idx), max(sampled_node_idx, new_node_idx)), None):
            row.append(sampled_node_idx)
            col.append(new_node_idx)
            cache[(min(sampled_node_idx, new_node_idx), max(sampled_node_idx, new_node_idx))] = True
            total_add += 1
    row = torch.IntTensor(row)
    col = torch.IntTensor(col)
    edge_index_to_add = torch.stack([row, col], dim = 0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim = 1)
    edge_index = to_undirected(edge_index)
    data.edge_index = edge_index
    return data
    

def bincount2d(arr, bins=6):
    ## bins corresponds to the max label class
    count = torch.zeros((arr.shape[0], bins), dtype=torch.int)
    indexing = (torch.ones_like(arr).T * torch.arange(len(arr))).T
    one_mat = torch.ones_like(arr, dtype=torch.int)
    count.index_put_((indexing, arr), one_mat, accumulate=True)
    return count


def find_neighbor_hist(edge_index, node, labels):
    row, _ = edge_index[0], edge_index[1]
    edge_index = remove_self_loops(edge_index)[0]
    adj = to_dense_adj(edge_index).squeeze(dim = 0)
    node_neighbor = adj[node]
    max_label_plus = labels.max().item() + 2
    index_mat = torch.repeat_interleave(torch.arange(node_neighbor.shape[1]).reshape(1, -1), node_neighbor.shape[0], dim = 0)
    neigh_pos = torch.where(node_neighbor != 0, labels[index_mat], max_label_plus - 1)
    hist = bincount2d(neigh_pos, bins=max_label_plus)
    hist = hist[:, :-1]
    return hist

    


def cross_class_neighborhood_similarity(data, c, c_prime):
    """
        CCNS, def 2 of Yao's paper
    """
    node_label_cat = category_node_by_label(data)
    score = 0
    class_c_nodes = node_label_cat[c]
    class_c_prime_nodes = node_label_cat[c_prime]
    edge_index = data.edge_index
    #num_of_labels = int(data.y.max()) + 1
    # cosine_similarity = torch.nn.CosineSimilarity(dim = 0).cuda()
    cc = find_neighbor_hist(edge_index, class_c_nodes, data.y).float()
    # ipdb.set_trace()
    ccp = find_neighbor_hist(edge_index, class_c_prime_nodes, data.y).float()

    cc = F.normalize(cc, p = 2., dim = 1)
    ccp = F.normalize(ccp, p = 2, dim = 1)

    score = torch.matmul(cc, ccp.T)
    return score.sum() / len(class_c_nodes) / len(class_c_prime_nodes)

    


def heatmap(score, filename = 'sanity_check.png'):
    l = len(score)
    df = pd.DataFrame(columns=['c', 'c_prime', 'score'])
    for i in range(l):
        for j in range(l):
            s = float(score[i][j])
            df.loc[len(df.index)] = [i, j, s]
    df.to_csv('test.csv')
    img = df.pivot("c", "c_prime", "score")
    sns.heatmap(img, annot=True)
    plt.savefig(filename)
    plt.clf()

def new_edge_addition(data, K, direction = 'homo->hete', mask='test', ratio = 0.05, dataset = 'Cora', pos = 'middle'):
    if mask == 'test':
        if dataset == 'Cora':
            mask_selected = data.test_masks[0]
        else:
            mask_selected = data.test_mask[:, 0]
    else:
        if dataset == 'Cora':
            mask_selected = data.train_masks[0]
        else:
            mask_selected = data.test_mask[:, 0]
    edge_index = data.edge_index
    device = edge_index.device
    row = []
    col = []
    node_idxs = np.arange(data.x.shape[0])
    node_homo_res, _, _ = compute_homo_mask_new(data, None, False)
    node_homo_res = node_homo_res.numpy()
    total_class_num = data.y.max().item() + 1
    total_node_idx = defaultdict(list)
    if direction == 'homo->hete':
        node_sorted_idx = np.argsort(node_homo_res)
    else:
        node_sorted_idx = np.argsort(node_homo_res)[::-1]
    ## get 0.05 for each class
    total_mask_num = data.x[mask_selected].shape[0]
    req_num = int(total_mask_num / total_class_num * ratio)
    req_arr = np.zeros(total_class_num)
    req_to_be = total_class_num
    if pos == 'middle':
        start = len(node_sorted_idx) // 2 - 1
    else:
        start = len(node_sorted_idx) - 1
    masked_nodes = node_idxs[mask_selected]
    for i in range(start, -1, -1):
        idx = node_sorted_idx[i]
        corr_label = data.y[idx].item()
        if idx not in masked_nodes:
            continue
        if req_arr.min() >= req_num:
            break
        if req_arr[corr_label] >= req_num: 
            continue
        else:
            req_arr[corr_label] += 1
            total_node_idx[corr_label].append(idx.item())
    label_list = np.arange(total_class_num)
    row = []
    col = []
    G = to_networkx(data, to_undirected = True, remove_self_loops = True)
    add_num = 0
    threshold = 3 * K
    total_round = 0
    while add_num < K and total_round < threshold:
        if direction == 'homo->hete':
            ## randomly pick two nodes
            start = np.random.choice(label_list, size = 1, replace = False)[0]
            rest_label_list = [x for x in label_list if x != start]
            end = np.random.choice(rest_label_list, size = 1, replace = False)[0]
        else:
            start = end = np.random.choice(label_list, size = 1, replace = False)[0]
        # import ipdb;ipdb.set_trace()
        start_node_idx = np.random.choice(total_node_idx[start], size = 1, replace = False)[0]
        end_node_idx = np.random.choice(total_node_idx[end], size = 1, replace = False)[0]
        if not G.has_edge(start_node_idx, end_node_idx):
            row.append(start_node_idx)
            col.append(end_node_idx)
            add_num += 1
        total_round += 1
    print(f"Add edge num: {add_num}")
    row = torch.IntTensor(row)
    col = torch.IntTensor(col)
    # import ipdb; ipdb.set_trace()
    edge_index_to_add = torch.stack([row, col], dim = 0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim = 1)
    edge_index = to_undirected(edge_index)
    data.edge_index = edge_index
    data.original_index = total_node_idx
    # ipdb.set_trace()
    return data    


def save_pyg_data(pyg_data, K, ratio, dataset, mask, path="/mnt/home/haitaoma/Graph-smooth/code/synthetic_data/pickle"):
    filename = osp.join(path, f"{K}-{ratio}-{dataset}-{mask}")
    save_best(pyg_data, filename)






def quick_sanity_check(part1 = False):
    ## Part 1: OK
    if part1: 
        args = generate_args()
        dataset, data = get_dataset(args)
        edge_num = data.edge_index.shape[1] // 2
        hetero_edge_addition(data, 12)
        new_edge_num = data.edge_index.shape[1] // 2
        assert edge_num + 12 == new_edge_num
        ## test homophily ratio, check table 3
        ## close to result in table 3
        dataset, data = get_dataset(args)
        edge_add = [1003, 1003, 1003, 2006, 2006, 2006, 2006, 4012, 4012, 4012]
        edge_add = [x // 2 for x in edge_add]
        for K in edge_add:
            data = hetero_edge_addition(data, K)
            print(homophily(data.edge_index, data.y))

    ## Part 2:
    # args = generate_args()
    # dataset, data = get_dataset(args)
    # ## compare to fig 4
    # ## to h = 0.25
    # data = hetero_edge_addition(data, K = 12036)
    # print(homophily(data.edge_index, data.y))
    # # data = data.cuda()
    # cross_score = torch.zeros((7, 7))
    # for i in tqdm.tqdm(range(7)):
    #     for j in tqdm.tqdm(range(7)):
    #         cross_score[i][j] = cross_class_neighborhood_similarity(data, i, j)
    # heatmap(cross_score)

    ## with noise
    # args = generate_args()
    # dataset, data = get_dataset(args)
    # data = hetero_edge_addition(data, K = 12036, with_noise = True, noise_level = 1.)
    # cross_score = torch.zeros((7, 7))
    # for i in tqdm.tqdm(range(7)):
    #     for j in tqdm.tqdm(range(7)):
    #         cross_score[i][j] = cross_class_neighborhood_similarity(data, i, j)
    # heatmap(cross_score, 'noise_1.0.png')

    ## ogb
    # args = generate_args()
    # vars(args)['dataset'] = 'arxiv'
    # dataset, data = get_dataset(args)
    # cross_score = torch.zeros((40, 40))
    # data.y = data.y.reshape(-1)
    # for i in tqdm.tqdm(range(40)):
    #     for j in tqdm.tqdm(range(40)):
    #         cross_score[i][j] = cross_class_neighborhood_similarity(data, i, j)
    # heatmap(cross_score, 'arxiv.png')



if __name__ == '__main__':
    quick_sanity_check()