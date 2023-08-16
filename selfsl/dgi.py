import torch.nn as nn
from .discriminator import Discriminator
from .gnn_encoder import AvgReadout
import numpy as np
import torch
import torch.nn.functional as F
from reparam_module import ReparamModule
from deeprobust.graph import utils


class DGI(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(DGI, self).__init__()
        self.gcn = encoder
        self.data = data
        self.processed_data = processed_data
        self.device = device

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        # self.mlp = MLP(nhid, 2*nhid)
        self.disc = Discriminator(nhid)
        self.b_xent = nn.BCEWithLogitsLoss()

        pseudo_labels = self.get_label()
        self.pseudo_labels = pseudo_labels.to(device)

    def get_label(self):
        nb_nodes = self.processed_data.features.shape[0]
        lbl_1 = torch.ones(nb_nodes, 2)
        lbl_2 = torch.zeros(nb_nodes, 2)
        lbl = torch.cat((lbl_1, lbl_2), dim=0)
        return lbl

    def make_loss(self, x):
        features = self.processed_data.features
        adj = self.processed_data.adj_norm
        nb_nodes = features.shape[0]

        self.train()
        idx = np.random.permutation(nb_nodes)

        shuf_fts = features[idx, :]

        logits = self.forward(features, shuf_fts, adj, None, None, None, None, embedding=x)
        loss = self.b_xent(logits, self.pseudo_labels)
        loss2 = self.softmax_entropy(logits).mean(0)
        # print('Loss:', loss.item())
        return loss, loss2

    def softmax_entropy(self, x):
        probabilities = F.softmax(x, dim=1)
        log_probabilities = F.log_softmax(x, dim=1)
        entropy_val = -(probabilities * log_probabilities).sum(dim=1)
        return entropy_val

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, embedding=None):
        if embedding is None:
            h_1 = self.gcn(seq1, adj)
        else:
            h_1 = embedding
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    # def fit(self):
    #     cnt_wait = 0
    #     best = 1e9
    #     best_t = 0
    #
    #     for epoch in range(nb_epochs):
    #         model.train()
    #         optimiser.zero_grad()
    #         idx = np.random.permutation(nb_nodes)
    #         shuf_fts = features[:, idx, :]
    #
    #         lbl_1 = torch.ones(batch_size, nb_nodes)
    #         lbl_2 = torch.zeros(batch_size, nb_nodes)
    #         lbl = torch.cat((lbl_1, lbl_2), 1)
    #
    #         if torch.cuda.is_available():
    #             shuf_fts = shuf_fts.cuda()
    #             lbl = lbl.cuda()
    #
    #         logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
    #         loss = b_xent(logits, lbl)
    #         print('Loss:', loss)



class DGISample(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(DGISample, self).__init__()
        self.gcn = encoder
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(nhid)
        self.b_xent = nn.BCEWithLogitsLoss()

        if kwargs['args'].dataset in ['reddit', 'arxiv']:
            self.sample_size = 2000
        else:
            self.sample_size = 2000
        self.pseudo_labels, self.test_pseudo_labels = self.get_label()
        self.b_xent = nn.BCEWithLogitsLoss()
        self.num_nodes = data.adj.shape[0]

    def get_label(self):
        lbl_1 = torch.ones(self.sample_size, 2)
        lbl_2 = torch.zeros(self.sample_size, 2)
        lbl = torch.cat((lbl_1, lbl_2))

        lbl_3 = torch.ones(500, 2)
        lbl_4 = torch.zeros(500, 2)
        lbl1 = torch.cat((lbl_3, lbl_4))

        return lbl.to(self.device), lbl1.to(self.device)

    def make_loss(self, x, params=None):
        features = self.processed_data.features
        adj = self.processed_data.adj_norm
        nb_nodes = features.shape[0]

        self.train()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]

        logits, test_logits = self.forward(features, shuf_fts, adj, None, None, None, None, params=params)
        # output = F.log_softmax(logits, dim=1)
        loss = self.b_xent(logits, self.pseudo_labels)
        # loss2 = self.softmax_entropy(logits).mean(0)
        loss2 = 0
        acc = self.get_accuracy(test_logits, self.test_pseudo_labels)
        # print('Loss:', loss.item())
        return loss, loss2, acc
        # print('Loss:', loss.item())

    def softmax_entropy(self, x):
        probabilities = F.softmax(x, dim=1)
        log_probabilities = F.log_softmax(x, dim=1)
        entropy_val = -(probabilities * log_probabilities).sum(dim=1)
        return entropy_val

    def get_accuracy(self, x, y):
        preds = torch.argmax(x, dim=1)
        # y = torch.argmax(label, dim=1)
        y = y[:, 0]

        acc = torch.sum(preds == y).float() / y.shape[0]
        return acc

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, params=None):
        idx = np.random.default_rng().choice(self.num_nodes, self.sample_size, replace=False)
        all_index = np.arange(self.num_nodes)
        test_idx = np.setdiff1d(all_index, idx)
        test_idx = np.random.choice(test_idx, 500, replace=False)
        # TODO: remove sparse
        if params is None:
            h_1 = self.gcn(seq1, adj)[idx]
            c = self.read(h_1, msk)
            c = self.sigm(c)
            h_2 = self.gcn(seq2, adj)[idx]

            h_3 = self.gcn(seq1, adj)[test_idx]
            d = self.read(h_3, msk)
            d = self.sigm(d)
            h_4 = self.gcn(seq2, adj)[test_idx]
        else:
            h_1 = self.gcn(seq1, adj, flat_param=params)[idx]
            c = self.read(h_1, msk)
            c = self.sigm(c)
            h_2 = self.gcn(seq2, adj, flat_param=params)[idx]

            h_3 = self.gcn(seq1, adj, flat_param=params)[test_idx]
            d = self.read(h_3, msk)
            d = self.sigm(d)
            h_4 = self.gcn(seq2, adj, flat_param=params)[test_idx]

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        ret1 = self.disc(d, h_3, h_4, samp_bias1, samp_bias2)
        return ret, ret1

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()




