# import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import collections

from ica.utils import load_data, pick_aggregator, create_map, build_graph
from ica.classifiers import LocalClassifier, RelationalClassifier, ICA
from scipy.stats import sem
from sklearn.metrics import accuracy_score
import sklearn
from utils import row_normalize
# from sklearn_extra.cluster import KMedoids
from utils import get_BALD_acquisition

import numpy as np
# import numba
# from numba import njit
import os
import time



def encode_onehot(labels):
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]

    return onehot_mx

class ICAAgent:   # 通过迭代更新特征信息为无label数据赋予label，直到label收敛或达到迭代次数

    def __init__(self, adj, features, labels, idx_train, idx_test, args):
        """
        idx_train: labeled data
        idx_test: unlabeled data
        """

        self.args = args
        self.adj = adj.tocsr()
        if args.dataset != 'reddit':
            self.adj_two_hop = adj.dot(adj)
            self.adj_two_hop.setdiag(0)
            self.adj_two_hop.eliminate_zeros()

        # self.graph = nx.from_scipy_sparse_matrix(adj)
        self.pseudo_labels = np.zeros((adj.shape[0], labels.shape[1]))
        # load_data = os.path.exists(f'preds/ICA_probs_{args.dataset}_{args.seed}.npy')
        load_data = os.path.exists(f'preds/ICA_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
        print('if loading: ', load_data)
        if not load_data:
            st = time.time()
            if args.dataset != 'cora':
                features[features!=0] = 1
            classifier = 'sklearn.linear_model.LogisticRegression'
            aggregate = 'count' # choices=['count', 'prop'], help='Aggregation operator'

            graph, domain_labels = build_graph(adj, features, labels)
            y_true = [graph.node_list[t].label for t in idx_test]
            local_clf = LocalClassifier(classifier)
            agg = pick_aggregator(aggregate, domain_labels)
            relational_clf = RelationalClassifier(classifier, agg)
            ica = ICA(local_clf, relational_clf, bootstrap=True, max_iteration=10)

            ica.fit(graph, idx_train)
            conditional_node_to_label_map = create_map(graph, idx_train)

            eval_idx = np.setdiff1d(range(adj.shape[0]), idx_train)
            ica_predict, probs = ica.predict(graph, eval_idx, idx_test, conditional_node_to_label_map)
            ica_accuracy = accuracy_score(y_true, ica_predict)
            print('Acc: ' + str(ica_accuracy))
            print('optimization consumes %s s' % (time.time()-st))
            # self.ica_predict = np.array([int(x[1:]) for x in ica_predict])
            dict_pred = {x: int(y[1:]) for x, y in zip(idx_test, ica_predict)}
            dict_train = {x: labels.argmax(1)[x] for x in idx_train}
            dict_pred.update(dict_train)
            concated = sorted(dict_pred.items(), key=lambda x: x[0])

            self.probs = np.vstack((labels[idx_train], probs))
            self.concated = np.array([y for x, y in concated])

            # np.save(f'preds/ICA_probs_{args.train_size}_{args.dataset}_{args.seed}.npy', self.probs)
            # np.save(f'preds/ICA_preds_{args.train_size}_{args.dataset}_{args.seed}.npy', self.concated)

        else:
            print('loading probs/preds...')
            # self.probs = np.load(f'ICA_probs_{args.dataset}_{args.seed}.npy')
            # self.concated = np.load(f'ICA_preds_{args.dataset}_{args.seed}.npy')
            # self.probs = np.load(f'ICA_probs_{args.dataset}_10.npy')
            # self.concated = np.load(f'ICA_preds_{args.dataset}_10.npy')

            self.probs = np.load(f'preds/ICA_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
            self.concated = np.load(f'preds/ICA_preds_{args.train_size}_{args.dataset}_{args.seed}.npy')

            # self.probs = np.load(f'preds/{args.dataset}_{args.seed}_pred.npy')
            # self.concated = self.probs.argmax(1)
            # self.concated[idx_train] = labels.argmax(1)[idx_train]
            print('Acc: ', (self.concated == labels.argmax(1))[idx_test].sum()/len(idx_test))

    def get_label(self, concated=None, use_probs=False):
        '''
        Get neighbor label distribution
        '''
        if use_probs:
            pred = self.probs
        else:
            if concated is None:
                concated = self.concated
            pred = encode_onehot(concated)

        A = self.adj

        st = time.time()
        if self.args.dataset != 'reddit':
            B = self.adj_two_hop
            self.pseudo_labels = _get_subgraph_label(pred, self.pseudo_labels, A.indptr, A.indices, B.indptr, B.indices)
        else:
            self.pseudo_labels = _get_neighbor_label(pred, self.pseudo_labels, A.indptr, A.indices)
        print('building label consumes %s s' % (time.time()-st))
        return torch.FloatTensor(self.pseudo_labels)


class LPAgent:

    def __init__(self, adj, features, labels, idx_train, idx_test, args):
        """
        :param graph: Networkx Graph.
        """

        self.adj = adj.tocsr()
        self.adj_two_hop = adj.dot(adj)
        self.adj_two_hop.setdiag(0)
        self.adj_two_hop.eliminate_zeros()
        self.pseudo_labels = np.zeros((adj.shape[0], labels.shape[1]))

        # load_data = False
        load_data = os.path.exists(f'preds/LP_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
        if not load_data:
            self.graph = nx.from_scipy_sparse_matrix(adj)
            lp_predict = self.hmn(labels.argmax(1), idx_train)
            # lp_predict = self.propagate(adj, labels.argmax(1), idx_train)
            lp_accuracy = accuracy_score(labels.argmax(1)[idx_test], lp_predict[idx_test])
            print('Acc: ' + str(lp_accuracy))
            dict_pred = {x: y for x, y in enumerate(lp_predict)}
            concated = sorted(dict_pred.items(), key=lambda x: x[0])
            self.concated = np.array([y for x, y in concated])

            # np.save(f'LP_probs_{args.dataset}_{args.seed}.npy', self.probs)
            # np.save(f'LP_preds_{args.dataset}_{args.seed}.npy', self.concated)
        else:
            # self.probs = np.load(f'LP_probs_{args.dataset}_{args.seed}.npy')
            # self.concated = np.load(f'LP_preds_{args.dataset}_{args.seed}.npy')
            # self.probs = np.load(f'LP_probs_{args.dataset}_10.npy')
            # self.concated = np.load(f'LP_preds_{args.dataset}_10.npy')
            self.probs = np.load(f'preds/LP_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
            self.concated = np.load(f'preds/LP_preds_{args.train_size}_{args.dataset}_{args.seed}.npy')

            print('Acc: ', (self.concated == labels.argmax(1))[idx_test].sum()/len(idx_test))

    def hmn(self, labels, idx_train):
        from networkx.algorithms import node_classification
        # import node_classification
        for id in idx_train:
            self.graph.nodes[id]['label'] = labels[id]
        preds = node_classification.harmonic_function(self.graph)
        return np.array(preds)

    def propagate(self, adj, labels, idx_train):
        # row_sums = adj.sum(axis=1).A1
        # row_sum_diag_mat = np.diag(row_sums)
        # adj_rw = np.linalg.inv(row_sum_diag_mat).dot(adj)
        adj_rw = row_normalize(self.adj.asfptype())
        Y = np.zeros(labels.shape)
        for id in idx_train:
            Y[id] = labels[id]

        for i in range(0, 1000):
            Y = adj_rw.dot(Y)
            for id in idx_train:
                Y[id] = labels[id]  # Clamping

        return Y.round()

    def get_label(self, concated=None):
        '''
        Get neighbor label distribution
        '''
        if concated is None:
            concated = self.concated
        pred = encode_onehot(concated)
        A = self.adj
        B = self.adj_two_hop
        st = time.time()
        # self.pseudo_labels = _get_neighbor_label(pred, self.pseudo_labels, A.indptr, A.indices)
        self.pseudo_labels = _get_subgraph_label(pred, self.pseudo_labels, A.indptr, A.indices, B.indptr, B.indices)
        print('building label consumes %s s' % (time.time()-st))

        return torch.FloatTensor(self.pseudo_labels)


class CombinedAgent(ICAAgent):

    def __init__(self, adj, features, labels, idx_train, idx_test, args):
        """
        :param graph: Networkx Graph.
        """

        self.adj = adj
        self.args=args
        unlabeled = np.array([x for x in range(adj.shape[0]) if x not in idx_train])

        probs = np.zeros(labels.shape)
        # LP
        agent = LPAgent(adj, features, labels, idx_train, unlabeled, args)
        preds = agent.concated
        probs += agent.probs

        # #FeatLP

        # agent = FeatLPAgent(adj, features, labels, idx_train, unlabeled, args)
        # preds = agent.concated
        # probs += agent.probs
        # import ipdb
        # ipdb.set_trace()

        # ICA
        agent = ICAAgent(adj, features, labels, idx_train, unlabeled, args)
        preds_ica = agent.concated
        probs += agent.probs

        # # GCN?
        # probs_gcn = np.load(f'preds/{args.dataset}_{args.seed}_pred.npy')
        # probs += probs_gcn
        # accuracy = accuracy_score(probs_gcn.argmax(1)[unlabeled], labels.argmax(1)[unlabeled])
        # print('Acc: ' + str(accuracy))

        final_preds = probs.argmax(1)
        accuracy = accuracy_score(final_preds[unlabeled], labels.argmax(1)[unlabeled])
        print('Acc: ' + str(accuracy))

        dict_pred = {x: y for x, y in enumerate(final_preds)}
        concated = sorted(dict_pred.items(), key=lambda x: x[0])
        self.concated = np.array([y for x, y in concated])
        self.pseudo_labels = np.zeros((adj.shape[0], labels.shape[1]))

        self.adj = adj.tocsr()
        self.adj_two_hop = adj.dot(adj)
        self.adj_two_hop.setdiag(0)
        self.adj_two_hop.eliminate_zeros()


def _get_neighbor_label(pred, pseudo_labels, iA, jA):
    '''
    Get neighbor label distribution
    '''
    for row in range(len(iA)-1):
        label_dist = pred[jA[iA[row]: iA[row+1]]].sum(0)
        pseudo_labels[row] = label_dist / label_dist.sum()
    return pseudo_labels

# @njit
def _get_subgraph_label(pred, pseudo_labels, iA, jA, iB, jB):
    '''
    Get neighbor label distribution
    '''
    for row in range(len(iA)-1):
        label_dist = pred[jA[iA[row]: iA[row+1]]].sum(0)
        label_dist += pred[jB[iB[row]: iB[row+1]]].sum(0)
        pseudo_labels[row] = label_dist / label_dist.sum()
    return pseudo_labels
