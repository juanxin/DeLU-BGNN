import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans
from ssl_utils import *
# from distance import *
from utils import get_BALD_acquisition
import os


class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features

class EdgeMask(Base):

    def __init__(self, adj, features, nhid, device):
        self.adj = adj
        self.masked_edges = None
        self.device = device
        self.features = features.to(device)
        self.cached_adj_norm = None
        self.pseudo_labels = None
        self.linear = nn.Linear(nhid, 2).to(device)

    def transform_data(self, mask_ratio=0.1):
        '''randomly mask edges'''
        # self.cached_adj_norm = None
        if self.cached_adj_norm is None:
            nnz = self.adj.nnz
            perm = np.random.permutation(nnz)
            preserve_nnz = int(nnz*(1 - mask_ratio))
            masked = perm[preserve_nnz: ]
            self.masked_edges = (self.adj.row[masked], self.adj.col[masked])
            perm = perm[:preserve_nnz]
            r_adj = sp.coo_matrix((self.adj.data[perm],
                                   (self.adj.row[perm],
                                    self.adj.col[perm])),
                                  shape=self.adj.shape)

            # renormalize_adj
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj
            # features = preprocess_features(self.features, self.device)
        return self.cached_adj_norm, self.features

    def make_loss(self, embeddings):
        '''link prediction loss'''
        edges = self.masked_edges
        # self.neg_edges = self.neg_sample(k=len(edges[0]))
        if self.pseudo_labels is None:
            self.pseudo_labels = np.zeros(2*len(edges[0]))
            self.pseudo_labels[: len(edges[0])] = 1
            self.pseudo_labels = torch.LongTensor(self.pseudo_labels).to(self.device)
            self.neg_edges = self.neg_sample(k=len(edges[0]))

        neg_edges = self.neg_edges
        node_pairs = np.hstack((np.array(edges), np.array(neg_edges).transpose()))
        self.node_pairs = node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, self.pseudo_labels)
        print("Edge prediction loss:{}".format(loss))
        # print(loss)
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels)
        # print(acc)
        # # return 0
        return loss

    def neg_sample(self, k):
        nonzero = set(zip(*self.adj.nonzero()))
        edges = self.random_sample_edges(self.adj, k, exclude=nonzero)
        return edges

    def random_sample_edges(self, adj, n, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))


def preprocess_features(features, device):
    return features.to(device)

def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
