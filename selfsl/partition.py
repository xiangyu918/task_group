import random

import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
from numba import njit
import networkx as nx
from sklearn.cluster import KMeans
import pymetis
import os
from deeprobust.graph import utils

class Par(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(Par, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.device = device
        if kwargs['args'].dataset in ['citeseer']:
            self.nparts = 1000
        elif kwargs['args'].dataset in ['photo', 'computers']:
            self.nparts = 100
        elif kwargs['args'].dataset in ['wiki']:
            self.nparts = 20
        else:
            self.nparts = 400
        pseudo_labels = self.get_label(self.nparts)
        self.pseudo_labels = pseudo_labels.to(device)
        # self.mlp = MLP(nhid, 2*nhid)
        self.disc = nn.Linear(nhid, self.nparts)
        self.sampled_indices = (self.pseudo_labels >= 0)  # dummy sampling
        self.sampled_size = 4000

    def make_loss(self, embeddings):
        # random.shuffle(self.sampled_indices)
        train_sampled_indices = self.sampled_indices[:self.sampled_size]
        test_sampled_indices = self.sampled_indices[self.sampled_size:]

        embeddings = self.disc(embeddings)
        train_embeddings = embeddings[:self.sampled_size]
        test_embeddings = embeddings[self.sampled_size:]
        train_pseudo_labels = self.pseudo_labels[:self.sampled_size]
        test_pseudo_labels = self.pseudo_labels[self.sampled_size:]

        output = F.log_softmax(train_embeddings, dim=1)
        loss = F.nll_loss(output[train_sampled_indices], train_pseudo_labels[train_sampled_indices])
        loss2 = self.softmax_entropy(embeddings[:self.sampled_size]).mean(0)
        acc = self.get_accuracy(test_embeddings[test_sampled_indices], test_pseudo_labels[test_sampled_indices])
        return loss, loss2, acc

    def get_label(self, nparts):
        partition_file = './saved/' + self.args.dataset + '_partition_%s.npy' % nparts
        if not os.path.exists(partition_file):
            print('Perform graph partitioning with Metis...')

            adj_coo = self.data.adj.tocoo()
            node_num = adj_coo.shape[0]
            adj_list = [[] for _ in range(node_num)]
            for i, j in zip(adj_coo.row, adj_coo.col):
                if i == j:
                    continue
                adj_list[i].append(j)

            _, partition_labels = pymetis.part_graph(nparts=nparts, adjacency=adj_list)
            np.save(partition_file, partition_labels)
            return torch.LongTensor(partition_labels)
        else:
            partition_labels = np.load(partition_file)
            return torch.LongTensor(partition_labels)

    def softmax_entropy(self, x):
        # return -(x.softmax(x) * x.log_softmax(1)).sum(1)
        probabilities = F.softmax(x, dim=1)
        log_probabilities = F.log_softmax(x, dim=1)
        entropy_val = -(probabilities * log_probabilities).sum(dim=1)
        return entropy_val

    def get_accuracy(self, x, y):
        preds = torch.argmax(x, dim=1)
        acc = torch.sum(preds == y).float() / y.shape[0]
        return acc

# class MLP(nn.Module):
#     def __init__(self, nfeat, nhid):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(nfeat, nhid)
#         self.fc2 = nn.Linear(nhid, nfeat)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, seq):
#         x = F.relu(self.fc1(seq))
#         ret = self.fc2(x)
#         return ret

