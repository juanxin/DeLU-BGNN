from __future__ import print_function
from __future__ import division

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.io import loadmat
import torch
from normalization import fetch_normalization, row_normalize
from pathlib import Path
import math
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
import os.path as osp
import random
IMBALANCE_THRESH = 101



# torch.cuda.device_count()


# seed = 5
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)



# util functions
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name,
                                                     transform=T.NormalizeFeatures())

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def myload_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]



    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally) - 500)
    idx_val = range(len(ally) - 500, len(ally))


    # idx_test = test_idx_range.tolist()


    # idx_train = range(len(y))


    # idx_val = range(len(y), len(y)+500)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, 1))


    idx_train = torch.LongTensor(np.where(train_mask)[0])
    idx_val = torch.LongTensor(np.where(val_mask)[0])
    idx_test = torch.LongTensor(np.where(test_mask)[0])

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_torch(mx):
    rowsum = torch.sum(mx, 1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_inv[torch.isnan(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.matmul(r_mat_inv, mx)
    return mx

def accuracy(output, labels):
    preds = output.argmax(1)
    labels = labels.cpu().numpy()
    correct = np.equal(preds, labels).astype(np.float32)

    correct = correct.sum()
    return correct / len(labels)

def accuracy_mrun(outputs, labels, inds):
    preds = outputs.max(2)[1].mode(0)[0].type_as(labels)
    correct = preds[inds].eq(labels[inds]).double()
    correct = correct.sum()
    return correct / len(inds)

def accuracy_mrun_np(outputs, labels_np, inds):
    preds_np = stats.mode(np.argmax(outputs, axis=2), axis=0)[0].reshape(-1).astype(np.int32)
    correct = np.equal(labels_np[inds], preds_np[inds]).astype(np.float32)
    correct = correct.sum()
    return correct / len(inds)

def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.cpu().numpy()
    y_true = encode_onehot(y_true)
    # y_pred = y_preds.cpu().detach().numpy()
    y_pred = y_preds
    return roc_auc_score(y_true, y_pred)


def f1_score(output, labels, inds):
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    preds_np = stats.mode(np.argmax(output, axis=2), axis=0)[0].reshape(-1).astype(np.int32)
    # pred_labels = output.argmax(1)
    # labels_2 = labels.cpu().numpy()
    return f1_score(labels[inds], preds_np[inds], average='macro')
    # try:
    #     from sklearn.metrics import f1_score
    # except ImportError:
    #     raise RuntimeError("This contrib module requires sklearn to be installed.")
    # print(output.shape)
    # print(output.max(1).shape)
    # pred_labels = output.max(1)
    # labels_2 = labels.cpu().numpy()
    # print(labels_2.shape)
    # return f1_score(labels_2, pred_labels, average='macro')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def uncertainty_mrun(outputs, labels, inds):
    softmax_mean = torch.exp(outputs).sum(0)/float(outputs.shape[0])
    log_softmax_mean = torch.log(softmax_mean + 1e-12)
    pred_entropy = -torch.sum(softmax_mean * log_softmax_mean,1)
    preds = outputs.max(2)[1].mode(0)[0].type_as(labels)
    entropies_list = pred_entropy[inds]
    preds_list = preds[inds]
    labels_list = labels[inds]

    num_thr = 10.
    thr = (entropies_list.max().data - entropies_list.min().data)/num_thr
    thr_start = entropies_list.min().data
    p_ac = [None]*int(num_thr + 1)
    p_ui = [None]*int(num_thr + 1)
    pavpu = [None]*int(num_thr + 1)
    for i in range(int(num_thr + 1)):
        tr = thr_start + thr*i
        mask_certain=(entropies_list < tr).double()
        accurate = preds_list.eq(labels_list).double()
        inaccurate = 1. - accurate
        ac = torch.sum(accurate*mask_certain).double()
        iu = torch.sum(inaccurate*(1.-mask_certain)).double()

        p_ac[i] = ac/torch.sum(mask_certain).double()
        p_ui[i] = iu/torch.sum(inaccurate).double()
        pavpu[i] = (ac+iu)/(torch.sum(mask_certain)+torch.sum(1.-mask_certain))

    return torch.stack(pavpu)

def get_BALD_acquisition(y_T):
    expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + 1e-10), axis=-1), axis=0)
    expected_p = np.mean(y_T, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
    return (entropy_expected_p - expected_entropy)

def sample_by_bald_difficulty(X, y_var, y, num_samples, y_T):


    BALD_acq = get_BALD_acquisition(y_T)
    p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
    p_norm = p_norm / np.sum(p_norm)
    indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
    X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices], "attention_mask": X["attention_mask"][indices]}
    y_s = y[indices]
    w_s = y_var[indices][:,0]
    return X_s, y_s, w_s


def sample_by_bald_easiness(X, y_var, y, num_samples, y_T):


    BALD_acq = get_BALD_acquisition(y_T)
    p_norm = np.maximum(np.zeros(len(BALD_acq)), (1. - BALD_acq)/np.sum(1. - BALD_acq))
    p_norm = p_norm / np.sum(p_norm)
    indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
    X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices], "attention_mask": X["attention_mask"][indices]}
    y_s = y[indices]
    w_s = y_var[indices][:,0]
    return X_s, y_s, w_s


def sample_by_bald_class_easiness(X, y_var, y, num_samples, num_classes, y_T):

    BALD_acq = get_BALD_acquisition(y_T)
    BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)

    samples_per_class = num_samples // num_classes
    X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = [], [], [], [], []
    for label in range(num_classes):
        X_input_ids, X_token_type_ids, X_attention_mask = X['input_ids'][y == label], X['token_type_ids'][y == label], X['attention_mask'][y == label]
        y_ = y[y==label]
        y_var_ = y_var[y == label]
        # p = y_mean[y == label]
        p_norm = BALD_acq[y==label]
        p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
        p_norm = p_norm/np.sum(p_norm)
        if len(X_input_ids) < samples_per_class:
            replace = True
        else:
            replace = False
        indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
        X_s_input_ids.extend(X_input_ids[indices])
        X_s_token_type_ids.extend(X_token_type_ids[indices])
        X_s_attention_mask.extend(X_attention_mask[indices])
        y_s.extend(y_[indices])
        w_s.extend(y_var_[indices][:,0])
    X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
    return {'input_ids': np.array(X_s_input_ids), 'token_type_ids': np.array(X_s_token_type_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)


def sample_by_bald_class_difficulty(X, y_var, y, num_samples, num_classes, y_T):

    BALD_acq = get_BALD_acquisition(y_T)
    samples_per_class = num_samples // num_classes
    X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = [], [], [], [], []
    for label in range(num_classes):
        X_input_ids, X_token_type_ids, X_attention_mask = X['input_ids'][y == label], X['token_type_ids'][y == label], X['attention_mask'][y == label]
        y_ = y[y==label]
        y_var_ = y_var[y == label]
        p_norm = BALD_acq[y==label]
        p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
        p_norm = p_norm/np.sum(p_norm)
        if len(X_input_ids) < samples_per_class:
            replace = True

        else:
            replace = False
        indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
        X_s_input_ids.extend(X_input_ids[indices])
        X_s_token_type_ids.extend(X_token_type_ids[indices])
        X_s_attention_mask.extend(X_attention_mask[indices])
        y_s.extend(y_[indices])
        w_s.extend(y_var_[indices][:,0])
    X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
    return {'input_ids': np.array(X_s_input_ids), 'token_type_ids': np.array(X_s_token_type_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def numpy_to_coosparese(numpy_array):
    new_adj_data = []
    new_adj_row = []
    new_adj_col = []
    for i in range(len(numpy_array[0])):
        for j in range(len(numpy_array[1])):
            if numpy_array[i][j] != 0:
                new_adj_data.append(1)
                new_adj_row.append(i)
                new_adj_col.append(j)

    adj = sp.coo_matrix((new_adj_data, (new_adj_row, new_adj_col)), shape=numpy_array.shape)
    return adj

def clone_list(li1):
    li_copy = li1[:]
    return li_copy


def full_load_data(dataset_name, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = full_load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join(r'C:\Users\隽歆\PycharmProjects\GDC-master\new_data', dataset_name, 'out1_graph_edges.txt')
        graph_adjacency_list_file_path = Path(graph_adjacency_list_file_path).as_posix()
        graph_node_features_and_labels_file_path = os.path.join(r'C:\Users\隽歆\PycharmProjects\GDC-master\new_data', dataset_name,
                                                                'out1_node_feature_label.txt')
        graph_node_features_and_labels_file_path = Path(graph_node_features_and_labels_file_path).as_posix()
        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)

    g = adj

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    # g = sys_normalized_adjacency(g)
    #
    # g = sparse_mx_to_torch_sparse_tensor(g)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(float)
    rowsum = (rowsum==0)*1+rowsum

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def load_data_Blog():#
    #--------------------
    #
    #--------------------
    mat = loadmat('C:/Users/隽歆/PycharmProjects/SelfTask-GNN-master-blogCatalog2/data/BlogCatalog/blogcatalog.mat')
    adj = mat['network']
    label = mat['group']

    embed = np.loadtxt('C:/Users/隽歆/PycharmProjects/SelfTask-GNN-master-blogCatalog2/data/BlogCatalog/blogcatalog.embeddings_64')
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]

    features = normalize(feature)
    labels = np.array(label.todense().argmax(axis=1)).squeeze()

    labels[labels>16] = labels[labels>16]-1

    print("change labels order, imbalanced classes to the end.")
    #ipdb.set_trace()
    labels = refine_label_order(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = adj.todense()

    adj = sp.coo_matrix(adj)
    #adj = torch.FloatTensor(np.array(adj.todense()))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat