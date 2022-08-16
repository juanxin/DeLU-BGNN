from __future__ import print_function
from __future__ import division

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import scipy
import math
import torch.nn.functional as F
import time
import random
from model import BBGDCGCN
from utils import numpy_to_coosparese, clone_list, roc_auc_compute_fn, f1_score, accuracy, full_load_data,load_data_Blog, get_dataset
from utils import myload_data, accuracy_mrun_np, normalize_torch, preprocess_citation, get_BALD_acquisition, split_genuine
from selfsl import *
import os.path as osp
import argparse
import heapq
import sklearn
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cora", help="The data set")
parser.add_argument('--alpha', type=float, default=1, help='alpha for label correction')
parser.add_argument('--train_size', type=int, default=0, help='if plot')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data', default='cora', help='dateset')
args = parser.parse_args()






# torch.cuda.device_count()

seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Load data
if args.data in {'chameleon', 'cornell', 'texas', 'wisconsin'}:
    datastr = args.data
    splitstr = r'C:\Users\隽歆\PycharmProjects\GDC-master\splits/'+args.data+'_split_0.6_0.2_'+'0'+'.npz'
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
elif args.data in {'BlogCatalog'}:
    adj, features, labels = load_data_Blog()
    idx_train, idx_val, idx_test, class_num_mat= split_genuine(labels)
elif args.data in {'Coauthor-CS', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code'}:
    path = osp.expanduser('~\datasets')
    path = osp.join(path, args.data)
    dataset = get_dataset(path, args.data)
    data = dataset[0]
    features = data.x
    labels = data.y
    edge_index = data.edge_index.numpy()

    edge_index_row = np.array(edge_index[0]).tolist()
    edge_index_col = np.array(edge_index[1]).tolist()

    labels_print = labels.numpy()

    labels_print_2 = []
    adj_row = edge_index[0]
    adj_colume = edge_index[1]
    adj_value = []
    for i in range(len(adj_row)):
        adj_value.append(1)
    adj = sp.coo_matrix((adj_value, (adj_row, adj_colume)))
    index_train = []
    index_val = []

    for i_label in range(data.y.max() + 1):
        index_sub = [i for i, x in enumerate(data.y) if x == i_label]  # train/val index
        index_sub = random.sample(index_sub, 60)
        index_train += index_sub[:30]
        index_val += index_sub[30:]

    # import ipdb;ipdb.set_trace()
    index_train.sort()
    index_val.sort()
    index_train_val = index_val + index_train
    index_test = [i for i in range(data.y.shape[0]) if i not in index_train_val]
    adj = adj
    features = features
    labels = labels.numpy().T.squeeze()
    labels = torch.from_numpy(labels)
    idx_train = torch.from_numpy(np.array(index_train))
    idx_val = torch.from_numpy(np.array(index_val))
    idx_test = torch.from_numpy(np.array(index_test))

else:
    adj, features, labels, idx_train, idx_val, idx_test = myload_data(args.data)




# hyper-parameters
labels_true = labels.clone()
nfeat = features.shape[1]
number_nodes = features.shape[0]
nclass = labels.max().item() + 1
nfeat_list = [nfeat, 128, 128, 128, nclass]
nlay = 4
nblock = 1
# nlay = 2
# nblock = 1
# nfeat_list = [nfeat, 128, nclass]
num_edges = int(adj.nnz/2)
print("number of edge:{}".format(num_edges))
# num_edges = int(adj.shape[0] ** 2)
dropout = 0
lr = 0.005
weight_decay = 5e-3
mul_type='norm_first'
best_acc_test = 0
best_auc_test = 0
best_f1_score_test = 0




line = []
slice_node = []
for i in range(nclass):
    line.append([])
    slice_node.append([])
label_index = []
for i in range(nclass):
    label_index.append(i)
number_minority = nclass//2
random_index = random.sample(label_index, number_minority)
for i in range(len(idx_train)):
    for j in range(nclass):
        if (labels[i]==j):
            line[j].append(i)
# for i in range(nclass):
#     print(len(line[i])/2708)
number_node_total = 0
line_percent = []
for i in range(nclass):
    print("第{}类包含的节点数量：{}".format(str(i+1), len(line[i])))
    print("------------------------------------------------------------")
    line_percent.append(len(line[i])/len(idx_train))
for i in range(nclass):
    number_node_total = number_node_total + len(line[i])
print("训练集中的节点数量：{}".format(number_node_total))
print("验证集中的节点数量：{}".format(len(idx_val)))
print("测试集中的节点数量：{}".format(len(idx_test)))
print("------------------------------------------------------------")
slice_fix = []
slice_node_list = []
slice_node_list_fix = []
idx_train_list = []
idx_train_list_pseudo = []
best_class_acc_test = []
best_class_f1_socre = []
for i in range(nclass):
    if i in random_index:
        if len(line[i]) >= 10:
            slice_node[i] = random.sample(line[i], 10)
        else:
            slice_node[i] = line[i]
    else:
        if len(line[i]) >= 20:
            slice_node[i] = random.sample(line[i], 20)
        else:
            slice_node[i] = line[i]
for i in range(nclass):
    slice_node_list = slice_node_list+slice_node[i]
    slice_node_list_fix = slice_node_list_fix + slice_node[i]
    idx_train_list = idx_train_list + line[i]
slice_node_list.sort()

for i in range(nclass):
    slice_fix.append(slice_node[i])
idx_train_old = idx_train.clone()
idx_train = torch.tensor(slice_node_list)

print("nclass: %d\tnfea:%d" % (nclass, nfeat))
# defining model

adj_2 = torch.FloatTensor(adj.todense())
adj_normt = normalize_torch(adj_2 + torch.eye(adj_2.shape[0]))
num_edges = torch.sum(adj_normt!=0)

model = BBGDCGCN(nfeat_list=nfeat_list
                 , dropout=dropout
                 , nblock=nblock
                 , nlay=nlay
                 , num_edges=num_edges)

# adj_ssl = adj.todense()
# adj_ssl = scipy.sparse.coo_matrix(adj_ssl)


adj_ssl, features_ssl = preprocess_citation(adj, features, normalization="NoNorm")
ssl_agent_2 = ICAContextLabel(adj_ssl, features, labels, nclass=nclass, idx_train=slice_node_list, nhid=128,
                                      device='cuda', args=args)
optimizer = torch.optim.Adam(list(model.parameters()) + list(ssl_agent_2.linear.parameters()) , lr=lr)

print("Model Summary:")
print(model)
print('----------------------')


adj_print = adj.todense()
degree = []
for i in range(nclass):
    degree.append([])
for i in range(number_nodes):
    degree[labels[i]].append(np.sum(adj_print[i, :]))
for i in range(nclass):
    print(np.mean(degree[i]))
for i in range(nclass):
    print(np.sum(degree[i]))



adj = torch.FloatTensor(adj.todense())
adj_normt = normalize_torch(adj + torch.eye(adj.shape[0]))

if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    labels_true = labels_true.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    adj = adj.cuda()
    adj_normt = adj_normt.cuda()


labels_np = labels.cpu().numpy().astype(np.int32)
labels_true_np = labels_true.cpu().numpy().astype(np.int32)
idx_train_np = idx_train.cpu().numpy()
idx_val_np = idx_val.cpu().numpy()
idx_test_np = idx_test.cpu().numpy()



# training

nepochs = 400
num_run = 20
# for i in range(100):
#     # t = time.time()
#     optimizer.zero_grad()
#     train_edg = True
#     wup = np.min([1.0, (i + 1) / 20])
#
#     train_adj, train_fea = ssl_agent.transform_data()
#
#     # wup = 1.0
#
#     output, output_2, embedding, tot_loss, nll_loss, kld_loss, drop_rates = model(
#                                                                         train_edg
#                                                                         ,x=features
#                                                                         , labels=labels
#                                                                         , adj=adj
#                                                                         , obs_idx=idx_train
#
#                                                                         , warm_up=wup
#                                                                         , adj_normt=adj_normt
#
#                                                                         , training=True
#                                                                         , mul_type=mul_type
#                                                                         )
#     loss_ssl = ssl_agent.make_loss(embedding)
#
#
#
#     main_loss = loss_ssl
#     main_loss.backward()
#     optimizer.step()
#
#     # outs = [None] * num_run
#     # for j in range(num_run):
#     #     outstmp, output_2, embedding, _, _, _, _ = model(x=features
#     #                                            , labels=labels
#     #                                            , adj=adj
#     #                                            , obs_idx=idx_train
#     #                                            , warm_up=wup
#     #                                            , adj_normt=adj_normt
#     #                                            , training=False
#     #                                            , samp_type='rel_ber'
#     #                                            , mul_type=mul_type)
#     #
#     #     outs[j] = outstmp.cpu().data.numpy()
#     #
#     # outs = np.stack(outs)
#     # acc_val_tr = accuracy_mrun_np(outs, labels_np, idx_val_np)
#     # acc_test_tr = accuracy_mrun_np(outs, labels_np, idx_test_np)
#
#     print('Epoch: {:04d}'.format(i + 1)
#           , 'loss_edge: {:.4f}'.format(loss_ssl.item()))
#           # , 'kld: {:.4f}'.format(kld_loss.item())
#           # , 'acc_val: {:.4f}'.format(acc_val_tr)
#           # , 'acc_test: {:.4f}'.format(acc_test_tr))
#     print('----------------------')
#     print('random minority label: {}'.format(random_index))
#     print('label percentage: {}'.format(line_percent))
#     print('----------------------')


for i in range(nepochs):
    # t = time.time()
    optimizer.zero_grad()
    wup = np.min([1.0, (i+1)/20])
    train_edg = False


    # wup = 1.0
    output, output_2, embedding, tot_loss, nll_loss, kld_loss, drop_rates = model(
                                                             train_edg
                                                             ,slice_node_list_fix
                                                             ,x=features
                                                             , labels=labels
                                                             , adj=adj
                                                             , obs_idx=idx_train
                                                             , warm_up=wup
                                                             , adj_normt=adj_normt
                                                             , training=True
                                                             , mul_type=mul_type
                                                            )
    # loss_ssl2 = ssl_agent_2.make_loss(embedding)

    l2_reg = None
    block_index = 0
    for layer in range(nlay):
        l2_reg_lay = None
        if layer==0:
            for param in model.gcs[str(block_index)].parameters():
                if l2_reg_lay is None:
                    l2_reg_lay = (param**2).sum()
                else:
                    l2_reg_lay = l2_reg_lay + (param**2).sum()
            block_index += 1

        else:
            for iii in range(nblock):
                for param in model.gcs[str(block_index)].parameters():
                    if l2_reg_lay is None:
                        l2_reg_lay = (param**2).sum()
                    else:
                        l2_reg_lay = l2_reg_lay + (param**2).sum()
                block_index += 1

        l2_reg_lay = (1-drop_rates[layer])*l2_reg_lay

        if l2_reg is None:
            l2_reg = l2_reg_lay
        else:
            l2_reg += l2_reg_lay

    # main_loss = tot_loss
    main_loss = tot_loss + weight_decay * l2_reg
    main_loss.backward()
    optimizer.step()

    outs = [None]*num_run
    outs_2 = [None] * num_run
    for j in range(num_run):
        outstmp, output_2, embedding, _, _, _,_ = model(
                                    train_edg
                                   ,slice_node_list_fix
                                   , x=features
                                   , labels=labels
                                   , adj=adj
                                   , obs_idx=idx_train
                                   , warm_up=wup
                                   , adj_normt=adj_normt
                                   , training=False
                                   , samp_type='rel_ber'
                                   , mul_type=mul_type
                                   )

        outs[j] = outstmp.cpu().data.numpy()

    outs = np.stack(outs)
    acc_val_tr = []
    acc_test_tr = []
    auc_test = []
    test_result_f1_score = []

    for j in range(num_run):
        acc_val_tr.append(accuracy(outs[j][idx_val_np], labels_true[idx_val_np]))
        acc_test_tr.append(accuracy(outs[j][idx_test_np], labels_true[idx_test_np]))
        auc_test.append(roc_auc_compute_fn(outs[j][idx_test_np], labels_true[idx_test_np]))
        # test_result_f1_score.append(f1_score(outs[j][idx_test_np], labels[idx_test_np]))
    acc_val_tr_2 = accuracy_mrun_np(outs, labels_true_np, idx_val_np)
    acc_test_tr_2 = accuracy_mrun_np(outs, labels_true_np, idx_test_np)
    # acc_val_tr = accuracy(outs[0][idx_val_np], labels[idx_val_np])
    # acc_test_tr = accuracy(outs[0][idx_test_np], labels[idx_test_np])
    # auc_test = roc_auc_compute_fn(outs[0][idx_test_np], labels[idx_test_np])
    test_result_f1_score_2 = f1_score(outs, labels_true_np, idx_test_np)

    class_node_list = []
    class_acc_test = []
    f1_score_list = []
    # if(acc_val_tr_2>best_acc_test):
    #     best_acc_test = acc_test_tr_2
    #     best_auc_test = sum(auc_test) / len(auc_test)
    #     best_f1_score_test = test_result_f1_score_2
    # if i>50:
    #
    #     TP = []
    #     FP = []
    #     FN = []
    #     precision = []
    #     recall = []
    #     for j in range(nclass):
    #         TP.append([])
    #         FP.append([])
    #         FN.append([])
    #     preds = output.max(1)[1].type_as(labels)
    #
    #     for m in range(nclass):
    #         for l in range(len(idx_test)):
    #             if not labels_true[idx_test[l]].eq(m):
    #                 if preds[idx_test[l]].eq(m):
    #                     FN[m].append(idx_test[l])
    #             if labels[idx_test[l]].eq(m):
    #                 if not preds[idx_test[l]].eq(m):
    #                     FP[m].append(idx_test[l])
    #                 if preds[idx_test[l]].eq(m):
    #                     TP[m].append(idx_test[l])
    #     test_f1score = 0
        # for j in range(nclass):
        #     if (len(TP[j]) + len(FP[j]))==0:
        #         precision.append(0)
        #     else:
        #         precision.append(len(TP[j]) / (len(TP[j]) + len(FP[j])))
        #     if (len(TP[j]) + len(FN[j]))==0:
        #         recall.append(0)
        #     else:
        #         recall.append(len(TP[j]) / (len(TP[j]) + len(FN[j])))
        #     if (precision[j] + recall[j]) ==0:
        #         f1_score_list.append(0)
        #     else:
        #         f1_score_list.append(2 * precision[j] * recall[j] / (precision[j] + recall[j]))
        #     test_f1score = test_f1score + f1_score_list[j]

        # for j in range(nclass):
        #     class_node_list.append([])
        # for j in range(len(idx_test)):
        #     class_node_list[labels_true[idx_test[j]]].append(idx_test[j])
        # for j in range(nclass):
        #     test_per_class = np.array(class_node_list[j]).astype(np.int64)
        #     # test_per_class.dtype = 'int64'
        #     class_acc_test.append(accuracy_mrun_np(outs, labels_true_np, test_per_class))
    if(acc_test_tr_2>best_acc_test):
        best_acc_test = acc_test_tr_2
        best_class_acc_test = class_acc_test
        best_class_f1_socre = f1_score_list
    if((sum(auc_test)/len(auc_test))>best_auc_test):
        best_auc_test = sum(auc_test)/len(auc_test)
    if(test_result_f1_score_2 > best_f1_score_test):
        best_f1_score_test = test_result_f1_score_2

    print('Epoch: {:04d}'.format(i+1)
          , 'nll: {:.4f}'.format(nll_loss.item())
          , 'kld: {:.4f}'.format(kld_loss.item())
          , 'acc_val: {:.4f}'.format(acc_val_tr_2)
          , 'acc_test: {:.4f}'.format(acc_test_tr_2)
          , 'auc_test: {:.4f}'.format(sum(auc_test)/len(auc_test))
          , 'f_score_test: {:.4f}'.format(test_result_f1_score_2))
    print('----------------------')
    print('random minority label: {}'.format(random_index))
    print('label percentage: {}'.format(line_percent))
    # print("------------------------------------------")
    # print(best_class_acc_test)
    # print("------------------------------------------")
    # print(best_class_f1_socre)


    if (i % 50 == 0) and i != 0:
        idx_train_list_pseudo = []

        ssl_agent_2 = ICAContextLabel(adj_ssl, features, labels, nclass=nclass, idx_train=slice_node_list, nhid=128,
                                      device='cuda', args=args)
        # ssl_agent_2 = LPContextLabel(adj_ssl, features, labels, nclass=nclass,
        #                              idx_train=slice_node_list, nhid=128, device='cuda', args=args)
        train_adj, train_fea = ssl_agent_2.transform_data()
        # loss_ssl = ssl_agent_2.make_loss()
        slice_list = []

        # slice_node_list = clone_list(slice_node_list_fix)
        for o in range(len(slice_fix)):
            slice_list.append([])

        for o in range(len(slice_fix)):
            slice_list[o] = slice_fix[o][:]

        output_bald = np.zeros((10, number_nodes, nclass))
        network_all_output_labels = np.zeros((number_nodes, nclass))
        for k in range(10):
            output, output_2, embedding, tot_loss, nll_loss, kld_loss, drop_rates = model(
                                                                                          train_edg
                                                                                          ,idx_train_list_pseudo
                                                                                          , x=features
                                                                                          , labels=labels
                                                                                          , adj=adj
                                                                                          , obs_idx=idx_train
                                                                                          , warm_up=wup
                                                                                          , adj_normt=adj_normt
                                                                                          , training=True
                                                                                          , mul_type=mul_type)
            output_bald[k] = output_2.cpu().detach().numpy()
            network_output_labels = output_2.argmax(dim=1)

            for j in range(number_nodes):
                network_all_output_labels[j][network_output_labels[j]] \
                    = network_all_output_labels[j][network_output_labels[j]] + 1

        network_output_labels = np.argmax(network_all_output_labels, axis=1)

        bald_value = get_BALD_acquisition(output_bald)
        pseudo_labels = ssl_agent_2.pseudo_labels.argmax(dim=1)
        m = 0
        n = 0

        for o in range(adj.shape[0]):
            # if o in idx_train_old:
                if o not in slice_node_list_fix:
                    if network_output_labels[o] in random_index:
                        if network_output_labels[o] == pseudo_labels[o]:
                            if bald_value[o] < 0.3:
                                idx_train_list_pseudo.append(o)
                                slice_node_list.append(o)
                                labels[o] = network_output_labels[o]
                                slice_list[pseudo_labels[o]].append(o)
                                m=m+1
                            if bald_value[o] < 0:
                                idx_train_list_pseudo.append(o)
                                slice_node_list.append(o)
                                labels[o] = network_output_labels[o]
                                slice_list[pseudo_labels[o]].append(o)
                                n=n+1
                        # else:
                        #     if bald_value[o] < 0:
                        #         slice_node_list.remove(o)
        print(m)
        print(n)
        slice_node_list = list(set(slice_node_list))
        idx_train = torch.tensor(slice_node_list)
        number_total = 0
        for o in range(len(slice_fix)):
            number_total = len(slice_list[o]) + number_total

        theta = 1 / nclass
        new_adj = adj.cpu().numpy()
        new_adj_2 = numpy_to_coosparese(new_adj)
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings_cos = embedding.cpu().detach().numpy()
        sim = cosine_similarity(embeddings_cos)
        add_edge_number = 0
        print(new_adj_2.nnz)
        line_percent_2 = []
        for o in range(len(slice_fix)):
            print(len(slice_list[o]) / number_total)
            line_percent_2.append(len(slice_list[o]) / number_total)
            if o in random_index:
                for j in range(len(slice_list[o])):
                    for p in range(adj.shape[0]):
                        if p != j:
                            if (sim[j][p] >= 0.95):
                                if new_adj[j][p] == 0:
                                    add_edge_number = add_edge_number + 1
                                    new_adj[j][p] = sim[j][p]
        print("此次总共加边数量为：{}".format(add_edge_number))
        adj = torch.from_numpy(new_adj).cuda()
        random_index = []
        line_percent = clone_list(line_percent_2)
        for o in range(number_minority):
            indexs = line_percent_2.index(min(line_percent_2))
            line_percent_2[indexs] = 1
            random_index.append(indexs)

    if i == 399:
        tsne = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
        embedding_cpu = embedding.cpu().detach().numpy()
        labels_cpu = labels.cpu()
        X_tsne = tsne.fit_transform(embedding_cpu)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(10, 10))

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_cpu)
        # for i in range(X_norm.shape[0]):
        #     plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()