import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
from numba import njit
import networkx as nx
import os
from deeprobust.graph import utils
from torch_geometric.utils import negative_sampling
from sklearn.neighbors import kneighbors_graph


class PairwiseAttrSim(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(PairwiseAttrSim, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.num_nodes = data.adj.shape[0]
        self.nclass = 2
        # self.mlp = MLP(nhid, 2*nhid)
        self.disc = nn.Linear(nhid, self.nclass)
        self.build_knn(self.data.features, k=10)

    def build_knn(self, X, k=10):
        args = self.args
        if not os.path.exists(f'saved/{args.dataset}_knn_{k}.npz'):
            A_knn = kneighbors_graph(X, k, mode='connectivity',
                            metric='cosine', include_self=True, n_jobs=4)
            print(f'saving saved/{args.dataset}_knn_{k}.npz')
            sp.save_npz(f'saved/{args.dataset}_knn_{k}.npz', A_knn)
        else:
            print(f'loading saved/{args.dataset}_knn_{k}.npz')
            A_knn = sp.load_npz(f'saved/{args.dataset}_knn_{k}.npz')
        self.edge_index_knn = torch.LongTensor(A_knn.nonzero())

    def sample(self, n_samples=4000):
        labels = []
        sampled_edges = []
        test_sampled_edges = []
        test_labels = []

        num_edges = self.edge_index_knn.shape[1]
        idx_selected = np.random.default_rng().choice(num_edges,
                        n_samples, replace=False).astype(np.int32)

        labels.append(torch.ones(len(idx_selected), dtype=torch.long))
        sampled_edges.append(self.edge_index_knn[:, idx_selected])

        neg_edges = negative_sampling(
                    edge_index=self.edge_index_knn, num_nodes=self.num_nodes,
                    num_neg_samples=n_samples)

        sampled_edges.append(neg_edges)
        labels.append(torch.zeros(neg_edges.shape[1], dtype=torch.long))

        labels = torch.cat(labels).to(self.device)
        sampled_edges = torch.cat(sampled_edges, axis=1)

        all_index = np.arange(num_edges)
        test_idx_selected = np.setdiff1d(all_index, idx_selected)
        test_idx_selected = np.random.choice(test_idx_selected, 1000, replace=False).astype(np.int32)
        test_labels.append(torch.ones(len(test_idx_selected), dtype=torch.long))
        test_sampled_edges.append(self.edge_index_knn[:, test_idx_selected])

        test_neg_edges = negative_sampling(
                         edge_index=self.edge_index_knn, num_nodes=self.num_nodes,
                         num_neg_samples=n_samples)
        test_sampled_edges.append(test_neg_edges)
        test_labels.append(torch.zeros(test_neg_edges.shape[1], dtype=torch.long))

        test_labels = torch.cat(test_labels).to(self.device)
        test_sampled_edges = torch.cat(test_sampled_edges, axis=1)

        return sampled_edges, labels, test_sampled_edges, test_labels

    def make_loss(self, embeddings):
        node_pairs, labels, test_node_pairs, test_labels = self.sample()
        # embs = self.mlp(embeddings)
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        test_embeddings0 = embeddings[test_node_pairs[0]]
        test_embeddings1 = embeddings[test_node_pairs[1]]

        embeddings = self.disc(torch.abs(embeddings0 - embeddings1))
        test_embeddings = self.disc(torch.abs(test_embeddings0 - test_embeddings1))

        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, labels)
        loss2 = self.softmax_entropy(embeddings).mean(0)
        acc = self.get_accuracy(test_embeddings, test_labels)
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
