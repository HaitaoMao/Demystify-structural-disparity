
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon, CoraFull, Actor, WikipediaNetwork, WebKB
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import index_to_mask, remove_self_loops, to_scipy_sparse_matrix, dense_to_sparse
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_sparse
import pickle as pkl
from torch_geometric.data import Data
import random
from torch_geometric.typing import Tensor, Optional, Union, Tuple
import pickle
from copy import deepcopy
import torch_geometric

def to_torch_coo_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
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


def set_seed_config(seed):
    # print('===========')
    # random seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_split(args, dataset, data, seed, index=None, sparse=False):
    # here, we have the proportion split, but we do not use it
    # index is utilized for multiple fixed split
    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None
    
    if args.is_synthetic:
        xdata = deepcopy(data)
        if 'Cora' in args.dataset:
            xdata.edge_index = data.edge_index.to(torch.int64)
            xdata.train_mask = data.train_masks[seed]
            xdata.val_mask = data.val_masks[seed]
            xdata.test_mask = data.test_masks[seed]
        else:
            xdata.edge_index = data.edge_index.to(torch.int64)
            # import ipdb; ipdb.set_trace()
            xdata.train_mask = data.train_mask[:, seed]
            xdata.val_mask = data.val_mask[:, seed]
            xdata.test_mask = data.test_mask[:, seed]
        return xdata, {}

    if args.dataset == 'arxiv':
        # only fix split
        split_idx = dataset.get_idx_split()
        # print(split_idx)
        data.y = torch.squeeze(data.y) ## add this for make y [num, 1] to [num]
        data.train_mask = index_to_mask(split_idx['train'], data.x.shape[0])
        data.test_mask = index_to_mask(split_idx['test'], data.x.shape[0]) ## add this for convenience
        data.val_mask = index_to_mask(split_idx['valid'], data.x.shape[0]) ## add this for convenience
        # Toread, 

        data.train_idx = split_idx['train'].to(data.train_mask.device)
        data.val_idx = split_idx['valid'].to(data.train_mask.device)
        data.test_idx = split_idx['test'].to(data.train_mask.device)
        # return data, split_idx

    if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        # fixed split is utilized by default
        train_num = 20
        if args.is_fix:
            data = data
        elif args.num_split:
            data = random_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, num=train_num)
        else:
            data = proportion_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, proportion=args.ratio_fix)
        # import ipdb; ipdb.set_trace()        
    elif args.dataset == "cs" or args.dataset == "physics":
        data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(f'random split {args.dataset} split {args.num_split}')

    elif args.dataset == "computers" or args.dataset == "photo":
        data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(f'random split {args.dataset} split {args.num_split}')

    elif args.dataset in ["Chameleon", "Squirrel"]:
        dataset = get_wiki_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        if args.is_fix:
            data = load_wiki_fix_split(data, args.dataset, seed)
        elif args.custom:
            data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed, tr_ratio=args.train_ratio, val_ratio=args.val_ratio)
        else:
            data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed)

    elif args.dataset in ["Cornell", "Texas", "Wisconsin"]:
        # TODO: add fix split for heterophily dataset
        dataset = get_WebKB_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed)

    elif args.dataset in ["Actor"]:
        dataset = get_Actor_dataset(args.dataset, args.normalize_features, transform=transform)
        # import ipdb; ipdb.set_trace()
        data = dataset[0]
        data.edge_index = data.edge_index
        if args.is_fix:
            data = load_actor_fix_split(data, args.dataset, seed)
        elif args.custom:
            data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed, tr_ratio=args.train_ratio, val_ratio=args.val_ratio)
        else:
            data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed)

    split_idx = {}
    
    if args.is_new and args.dataset in ["Cora", "CiteSeer", "PubMed", "photo", "cs"]:
        # 20 per class 30 per class all other node
        if not args.is_synthetic: 
            data = torch.load(f"./CPF/dataset/{args.dataset}.pt")
            if args.normalize_features:
                data.x /= torch.norm(data.x, dim=-1, keepdim=True)
            data.edge_index = data.edge_index.to(torch.int64)
            data.train_mask = data.train_masks[seed]
            data.val_mask = data.val_masks[seed]
            data.test_mask = data.test_masks[seed]
        else:
            proxy_data = torch.load(f"./CPF/dataset/{args.dataset}.pt")
            if args.normalize_features:
                data.x /= torch.norm(data.x, dim=-1, keepdim=True)
            data.edge_index = data.edge_index.to(torch.int64)
            data.train_mask = proxy_data.train_masks[seed]
            data.val_mask = proxy_data.val_masks[seed]
            data.test_mask = proxy_data.test_masks[seed]

        # import ipdb;ipdb.set_trace()
    
    if args.is_new and args.dataset in ["amazon_ratings"]:
        data = torch.load(f"./data/{args.dataset}.pt")
        data.x = torch.where(torch.isnan(data.x), 0, data.x)
        norm = torch.norm(data.x, dim=-1, keepdim=False)
        if args.normalize_features:
            data.x /= torch.unsqueeze(norm, -1)
        norm_mask = (norm != 0)
        data.edge_index = data.edge_index.to(torch.int64)
        data.edge_index, _ = torch_geometric.utils.subgraph(norm_mask, data.edge_index, relabel_nodes=True)
        data.x = data.x[norm_mask]
        data.y = data.y[norm_mask]
        data.train_mask = data.train_masks[seed][norm_mask]
        data.val_mask = data.val_masks[seed][norm_mask]
        data.test_mask = data.test_masks[seed][norm_mask]

    if args.dataset in ["genius", "twitch-gamer"]:
        data.edge_index = data.edge_index.to(torch.int64)
        data.train_mask = data.train_masks[seed]
        data.val_mask = data.val_masks[seed]
        data.test_mask = data.test_masks[seed]

    if args.dataset in ['IGB_tiny']:
        data.edge_index = data.edge_index.to(torch.int64)
        data.train_mask = data.train_mask
        data.val_mask = data.val_mask
        data.test_mask = data.test_mask
    
    if args.is_ood:
        data = torch.load(f"./data/ood_data_new/{args.dataset}.pt")
    elif args.is_iid:
        data = torch.load(f"./data/iid_data_new/{args.dataset}.pt")
    return data, split_idx


def compute_homo_mask(data, is_hete):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    label = F.one_hot(data.y, num_classes=num_labels)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    deg = torch.unsqueeze(deg, dim=-1)
    ego_matrix = deg * label

    neighor_matrix = torch_sparse.spmm(edge_index, edge_value, num_nodes, num_nodes, label)
    results = ego_matrix - neighor_matrix
    results = results[torch.arange(num_nodes), data.y]
    results = 1 - torch.div( results, torch.squeeze(deg))
    
    # import ipdb; ipdb.set_trace()
    mask_begin = results >= 0
    
    ends = [0.2, 0.4, 0.6, 0.8, 1.0]
    if is_hete:
        ends = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    masks = []
    for end in ends:
        mask_end = (results <= end)

        mask = mask_begin * mask_end
        masks.append(mask)
        mask_begin = (results >= end)
    
    data.homo_masks = masks
    data.homo_ratios = ends
    return data


def get_dataset(args, sparse=False, is_large_com=False, **kwargs):
    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None
    
    if is_large_com:
        transform = T.LargestConnectedComponents()

    if args.is_synthetic:
        if not args.is_synthetic_full:
            if args.dataset == 'Cora' and args.syn_mask == 'test':
                file_path = f"./synthetic_data/pickle/{args.change}-0.05-{args.dataset}-{args.syn_mask}"
            elif args.dataset == 'Cora' and args.syn_mask == 'train':
                file_path = f"./synthetic_data/pickle/{args.change}-1.0-{args.dataset}-{args.syn_mask}"
            elif args.dataset == 'Squirrel' and args.syn_mask == 'test':
                file_path = f"./synthetic_data/pickle/{args.change}-0.15-{args.dataset}-test"
            else:
                file_path = f"./synthetic_data/pickle/{args.change}-1.0-{args.dataset}-train"
            data = read_and_unpkl(file_path)
            dataset = None   
            return dataset, data 
        else:
            dataset = None 
            data = torch.load(args.dataset, map_location='cpu')
            return dataset, data

    if args.dataset == "arxiv":
        # import ipdb; ipdb.set_trace()
        dataset = get_ogbn_dataset("ogbn-arxiv", args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        dataset = get_planetoid_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset == "cs" or args.dataset == "physics":
        dataset = get_coauthor_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset == "computers" or args.dataset == "photo":
        dataset = get_amazon_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset in ["Cornell", "Texas", "Wisconsin"]:
        dataset = get_WebKB_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset in ["Actor"]:
        dataset = get_Actor_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
    
    elif args.dataset in ["Chameleon", "Squirrel"]:
        dataset = get_wiki_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
    elif args.dataset in ["roman_empire", "amazon_ratings"]:
        # here is just a place holder, we re upload the file in another file
        dataset = get_wiki_dataset("Chameleon", args.normalize_features, transform=transform)
        data = dataset[0]
    
    elif args.dataset in ["genius", "twitch-gamer"]:
        data = torch.load(f"./Non_Homophily_Large_Scale/new_data/{args.dataset}.pt")
        if args.normalize_features:
            data.x /= torch.norm(data.x, dim=-1, keepdim=True)
        dataset = None
    
    elif args.dataset in ["IGB_tiny"]:
        data = torch.load(f"./data/{args.dataset}.pt")
        if args.normalize_features:
            data.x /= torch.norm(data.x, dim=-1, keepdim=True)
        dataset = None   


    else:
        print("wrong")
        exit()
    
    # import ipdb;ipdb.set_trace()


    # if args.is_synthetic:
    #     data = f"./synthetic_data/pickle/{args.change}-{args.class_ratio}-{args.dataset}"
    #     data = read_and_unpkl(data)
        # import ipdb;ipdb.set_trace()
    return dataset, data

def get_transform(normalize_features, transform):
    # import ipdb; ipdb.set_trace()
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform

    return transform



def largest_connected_components(data, connection='weak'):
    import numpy as np
    import scipy.sparse as sp

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    num_components, component = sp.csgraph.connected_components(
        adj, connection=connection)

    if num_components <= 1:
        return data

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return data.subgraph(torch.from_numpy(subset).to(torch.bool))

def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_WebKB_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_Actor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Actor(path)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def get_wiki_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WikipediaNetwork(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_ogbn_dataset(name, normalize_features=True, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name, path)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def load_actor_fix_split(data, name, seed):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    masks = np.load(f"{path}/raw/film_split_0.6_0.2_{seed}.npz")
    data.train_mask = torch.tensor(masks['train_mask']) > 0
    data.val_mask = torch.tensor(masks['val_mask']) > 0
    data.test_mask = torch.tensor(masks['test_mask']) > 0
    
    return data


def load_wiki_fix_split(data, name, seed):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    name_dict = {"Squirrel": "squirrel", "Chameleon": "chameleon"}
    name2 = name_dict[name]
    masks = np.load(f"{path}/{name2}/geom_gcn/raw/{name2}_split_0.6_0.2_{seed}.npz")
    data.train_mask = torch.tensor(masks['train_mask']) > 0
    data.val_mask = torch.tensor(masks['val_mask']) > 0
    data.test_mask = torch.tensor(masks['test_mask']) > 0
    
    return data


def random_planetoid_splits(data, num_classes, seed, num):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:num] for i in indices], dim=0)
    # print('len(train)', len(train_index))
    rest_index = torch.cat([i[num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    return data

def proportion_planetoid_splits(data, num_classes, seed, proportion):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(1)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:int(proportion*len(i))] for i in indices], dim=0)
    rest_index = torch.cat([i[int(proportion*len(i)):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    print('len(train)', len(train_index))

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:int(0.5*len(rest_index))], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[int(0.5*len(rest_index)):], size=data.num_nodes)
    return data



def random_coauthor_amazon_splits(data, num_classes, seed):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def random_WebKB_splits(data, num_classes, seed, tr_ratio=0.48, val_ratio=0.32):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)
    train_index = torch.cat([i[:int((tr_ratio)*len(i))] for i in indices], dim=0)
    val_index = torch.cat([i[int(tr_ratio*len(i)):int((tr_ratio + val_ratio)*len(i))] for i in indices], dim=0)
    rest_index = torch.cat([i[int((tr_ratio + val_ratio)*len(i)):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    
    return data

# def mask_to_index(index, size):
#     all_idx = np.arange(size)
#     return all_idx[index]

# def index_to_mask(index, size):
#     mask = torch.zeros((size, ), dtype=torch.bool)
#     mask[index] = 1
#     return mask

# add for wiki dataset
'''
def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['corafull']:
        dataset = CoraFull(path)
    elif name in ['arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-'+name)
    elif name in ['cs', 'physics']:
        dataset = Coauthor(path, name)
    elif name in ['computers', 'photo']:
        dataset = Amazon(path, name)
    elif name in ['wiki']:
        dataset = WikiCS(root='data/')
        dataset.name = 'wiki'
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if not if_dpr:
        return dataset

    if name == 'wiki':
        # return Pyg2Dpr(dataset, multi_splits=True)
        data =  Pyg2Dpr(dataset, multi_splits=True)
        # data.idx_train, data.idx_val, data.idx_test = get_train_val_test_gcn(data.labels)
        return data

    else:
        return Pyg2Dpr(dataset)


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, multi_splits=False, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name

        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv': # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        # enable link prediction ....
        # self.enable_link_prediction(pyg_data)

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape
        if not multi_splits:
            if hasattr(pyg_data, 'train_mask'):
                # for fixed split
                # self.idx_train, self.idx_val, self.idx_test = get_train_val_test_gcn(self.labels)
                # We don't use fixed splits in this project...
                self.idx_train = mask_to_index(pyg_data.train_mask, n)
                self.idx_val = mask_to_index(pyg_data.val_mask, n)
                self.idx_test = mask_to_index(pyg_data.test_mask, n)
                self.name = 'Pyg2Dpr'
            else:
                try:
                    # for ogb
                    self.idx_train = splits['train']
                    self.idx_val = splits['valid']
                    self.idx_test = splits['test']
                    self.name = 'Pyg2Dpr'
                except:
                    # for other datasets
                    # self.idx_train, self.idx_val, self.idx_test = get_train_val_test_gcn(self.labels)
                    self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                            nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)
        else:
            # For wiki
            self.splits = self.load_splits(pyg_data)
            self.idx_train = self.splits['train'][0] # by default, it is from the first split
            self.idx_val = self.splits['val'][0]
            self.idx_test = self.splits['test'][0]
            self.name = 'Pyg2Dpr'

    def load_splits(self, data):
        splits = {'train': [], 'val': [], 'test': []}
        n = data.num_nodes
        for i in range(0, data.train_mask.shape[1]):
            train_mask = data.train_mask[:, i]
            val_mask = data.val_mask[:, i]
            if len(data.test_mask.shape) == 1:
                test_mask = data.test_mask
            else:
                test_mask = data.test_mask[:, i]
            idx_train = mask_to_index(train_mask, n)
            idx_val = mask_to_index(val_mask, n)
            idx_test = mask_to_index(test_mask, n)

            splits['train'].append(idx_train)
            splits['val'].append(idx_val)
            splits['test'].append(idx_test)
        return splits




'''


def save_best(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def pkl_and_write(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)
    return path

def read_and_unpkl(path):
    with open(path, 'rb') as f:
        res = pkl.load(f)
    return res    


def read_npz(out_t_dir):
    return torch.from_numpy(np.load(out_t_dir)["arr_0"])



## these two functions come from graphless

def idx_split(idx, ratio, seed=0):
    """
    randomly split idx into two portions with ratio% elements and (1 - ratio)% elements
    """
    set_seed_config(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """
    Args:
        The original setting was transductive. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)
        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18
    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,
        where as indices start directly with 'idx_' correspond to the node indices in the original graph
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1 : N1 + N2]
    obs_idx_test = obs_idx_all[N1 + N2 :]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def load_synthetic_dataset(homo_ratio, dataset='Cora', graph_num=0, mode = 'sitao'):
    if mode == 'sitao':
        base_dir = './synthetic_data'
        full_path = osp.join(base_dir, 'random', homo_ratio)
        adj = osp.join(full_path, f"adj_{homo_ratio}_{graph_num}.pt")
        label = osp.join(full_path, f"label_{homo_ratio}_{graph_num}.pt")
        feature_path = osp.join(base_dir, 'features', dataset)
        features = osp.join(feature_path, f"Cora_{graph_num}.pt")
        adj = torch.load(adj)
        adj = dense_to_sparse(adj.to_dense())
        l = torch.load(label)
        l = torch.argmax(l.to_dense(), dim = 1)
        x = torch.load(features)
        x = x.to_dense()
        return Data(x=x, edge_index=adj[0], edge_attr=adj[1], y=l)


def extre_mask(args, data):
    with open(f"./data/extreme_mask/{args.dataset}.txt", "rb") as f:
        data = pickle.load(f)
    data.train_mask = data["train_mask"]    
    data.val_mask = data["val_mask"]    
    data.test_mask = data["test_mask"]    

    return data
        

