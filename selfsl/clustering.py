import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
from numba import njit
import networkx as nx
from sklearn.cluster import KMeans
import os
from deeprobust.graph import utils


class Clu(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(Clu, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.device = device
        self.ncluster = 10
        self.pseudo_labels = self.get_label(self.ncluster)
        self.pseudo_labels = self.pseudo_labels.to(device)
        # self.mlp = MLP(nhid, 2*nhid)
        self.disc = nn.Linear(nhid, self.ncluster)
        self.sampled_indices = (self.pseudo_labels >= 0) # dummy sampling

    def make_loss(self, embeddings):
        embeddings = self.disc(embeddings)
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        loss2 = self.softmax_entropy(embeddings).mean(0)
        acc = self.get_accuracy(embeddings[self.sampled_indices],self.pseudo_labels[self.sampled_indices])
        return loss, loss2, acc

    def softmax_entropy(self, x):
        probabilities = F.softmax(x, dim=1)
        log_probabilities = F.log_softmax(x, dim=1)
        entropy_val = -(probabilities * log_probabilities).sum(dim=1)
        return entropy_val

    def get_accuracy(self, x, y):
        preds = torch.argmax(x, dim=1)
        acc = torch.sum(preds == y).float() / y.shape[0]
        return acc

    def get_label(self, ncluster):
        cluster_file = './saved/' + self.args.dataset + '_cluster_%s.npy' % ncluster
        if not os.path.exists(cluster_file):
            print('perform clustering with KMeans...')
            kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(self.data.features)
            cluster_labels = kmeans.labels_
            np.save(cluster_file, cluster_labels)
            return torch.LongTensor(cluster_labels)
        else:
            cluster_labels = np.load(cluster_file)
            return torch.LongTensor(cluster_labels)

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

