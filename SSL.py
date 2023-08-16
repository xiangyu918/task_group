import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from selfsl import *
from deeprobust.graph.utils import to_tensor, normalize_adj_tensor, accuracy
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (f1_score, roc_auc_score, adjusted_rand_score,
        accuracy_score, average_precision_score, v_measure_score)
from tqdm import tqdm

import torch.nn.functional as F
import utils
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import wandb
import os
from reparam_module import ReparamModule


class EmptyData:
    def __init__(self):
        self.adj_norm = None
        self.features = None
        self.labels = None
        self.train_pos_edge_index = None
        # self.train_neg_edge_index = None
        self.val_pos_edge_index = None
        self.val_neg_edge_index = None
        self.test_pos_edge_index = None
        self.test_neg_edge_index = None


class SSL(nn.Module):

    def __init__(self, data, encoder, set_of_ssl, save_dir, args, device='cuda', **kwargs):
        super(SSL, self).__init__()
        self.args = args
        self.encoder = encoder.to(device)
        self.data = data
        self.adj = data.adj
        self.features = data.features
        self.nums_node = data.nums_node
        self.device = device
        self.set_of_ssl = set_of_ssl
        self.n_tasks = len(set_of_ssl)
        self.weight = torch.ones(len(set_of_ssl)).to(device)
        self.weight.requires_grad = True
        self.ssl_agent = []
        self.optimizer = None
        self.processed_data = EmptyData()
        self.save_dir = save_dir
        self.trajectory = []
        self.setup_ssl(set_of_ssl)
        self.expert_trajectory = []

    def reset_parameters(self):
        # self.weight.data.fill_(1/(self.weight.size(1)+1))
        self.linear.weight.data.fill_(1/self.n_tasks)

    def set_weight(self, values):
        self.weight = torch.FloatTensor(values).to(self.device)

    def setup_ssl(self, set_of_ssl):
        # initialize them
        args = self.args
        self.process_data()
        params = list(self.encoder.parameters())
        # params = []
        self.trajectory.append([p.detach().cpu() for p in self.encoder.parameters()])
        for ix, ssl in enumerate(set_of_ssl):
            agent = eval(ssl)(data=self.data,
                    processed_data=self.processed_data,
                    encoder=self.encoder,
                    nhid=self.args.hidden,
                    device=self.device,
                    args=args).to(self.device)
            self.ssl_agent.append(agent)
            if agent.disc is not None:
                params = params + list(agent.disc.parameters())

            if hasattr(agent, 'gcn2'):
                params = params + list(agent.gcn2.parameters())
        params = params + [self.weight]
        # params = list(self.weight)
        self.optimizer = optim.Adam(params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        # print(params)

    def process_data(self):
        # still needed
        if self.processed_data.adj_norm is None:
            features, adj, labels = to_tensor(self.data.features, self.data.adj, self.data.labels, device=self.device)
            adj_norm = normalize_adj_tensor(adj, sparse=True)
            self.processed_data.adj_norm = adj_norm
            self.processed_data.features = features
            self.processed_data.labels = labels
            self.processed_data.train_pos_edge_index = self.data.train_pos_edge_index
            # self.train_neg_edge_index = self.data['train_neg_edge_index']
            self.processed_data.val_pos_edge_index = self.data.val_pos_edge_index
            self.processed_data.val_neg_edge_index = self.data.val_neg_edge_index
            self.processed_data.test_pos_edge_index = self.data.test_pos_edge_index
            self.processed_data.test_neg_edge_index = self.data.test_neg_edge_index
            # self.processed_data.edge_index = self.data.edge_index

    def pretrain(self, patience=1e5, filename=None, verbose=True):
        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm
        # features = self.data.x
        # adj_norm = self.data.edge_index
        if self.args.epochs == 0:
            with torch.no_grad():
                x = self.encoder(features, adj_norm)
            return x.detach()

        best_loss = 1e5
        pat = 0
        for i in range(self.args.epochs):
            weight_loss_dict = {}
            weight_dict = {}
            entropy_dict = {}
            self.optimizer.zero_grad()
            x = self.encoder(features, adj_norm)
            task_loss, loss_dict, entropy_loss, task_acc = self.get_task_loss(x)
            weight_task_loss = torch.mul(self.weight, task_loss)
            loss = torch.sum(weight_task_loss)
            for idx, (k, v) in enumerate(loss_dict.items()):
                weight_loss_dict[k + '_weight_loss'] = v * self.weight[idx]
                weight_dict[k + '_weight'] = self.weight[idx]
            for idx, (k, v) in enumerate(entropy_loss.items()):
                entropy_dict[k + '_entropy_loss'] = v

            log_dict = {k: v.item() for k, v in loss_dict.items()}
            log_dict.update({k: v.item() for k, v in entropy_dict.items()})
            log_dict.update({k: v.item() for k, v in weight_loss_dict.items()})
            log_dict.update({k: v.item() for k, v in weight_dict.items()})
            log_dict.update({"task_acc": task_acc})
            log_dict.update({"epoch": i})
            log_dict.update({'total_loss': loss})
            wandb.log(log_dict)
            # wandb.log({"epoch": i})
            # wandb.log({k: v.item() for k, v in loss_dict.items()})
            # wandb.log({k: v.item() for k, v in weight_loss_dict.items()})
            # wandb.log({k: v.item() for k, v in weight_dict.items()})
            if i == 0:
                initial_task_loss = task_loss.data.cpu()
                initial_task_loss = initial_task_loss.numpy()

            if i % 50 == 0 and verbose:
                print(f'Epoch {i}: {loss.item()}')
            if loss < best_loss:
                best_loss = loss
                best_weights = deepcopy(self.encoder.state_dict())
                pat = 0
            else:
                pat += 1
            if pat == patience:
                print('Early Stopped at Epoch %s' % i)
                break

            loss.backward(retain_graph=True)

            self.weight.grad.data = self.weight.grad.data * 0.0

            if self.args.mode == 'grad_norm':
                W = self.get_last_shared_layer()
                norms = []
                for idx, ssl in enumerate(self.ssl_agent):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(task_loss[idx], W[idx].parameters(), retain_graph=True)
                    # compute the norm
                    a = gygw[0]
                    b = self.weight[0]
                    norms.append(torch.norm(torch.mul(self.weight[idx], gygw[0])))
                norms = torch.stack(norms)
                # print('G_w(t): {}'.format(norms))

                # compute the inverse training rate r_i(t)
                # \curl{L}_i
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # print('r_i(t): {}'.format(inverse_train_rate))

                # compute the mean norm \tilde{G}_w(t)
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                # print('tilde G_w(t): {}'.format(mean_norm))

                # compute the GradNorm loss
                # this term has to remain constant
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.args.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                # print('Constant term: {}'.format(constant_term))
                # this is the GradNorm loss itself
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                self.weight.grad = torch.autograd.grad(grad_norm_loss, self.weight)[0]

                # do a step with the optimizer
            self.optimizer.step()

            # renormalize
            normalize_coeff = len(self.ssl_agent) / torch.sum(self.weight.data, dim=0)
            self.weight.data = self.weight.data * normalize_coeff

            self.trajectory.append([p.detach().cpu() for p in self.encoder.parameters()])

        # print("Saving {}".format(os.path.join(self.save_dir, filename)))
        # torch.save(self.trajectory, os.path.join(self.save_dir, filename))

        self.encoder.eval()
        self.encoder.load_state_dict(best_weights)
        # torch.save(self.encoder.state_dict(), "model/computers/PairwiseDistance.pth")
        with torch.no_grad():
            x = self.encoder(features, adj_norm)
        return x.detach()

    def matching(self, patience=200, file=None, verbose=True):
        lr = torch.tensor(0.01).to(self.device)
        # lr.requires_grad = True
        lr = lr.detach().to(self.device).requires_grad_(True)
        optimizer_lr = torch.optim.SGD([lr], lr=1e-5, momentum=0.5)
        # schedular_lr = torch.optim.lr_scheduler.StepLR(optimizer_lr, 200, 0.5)
        # self.encoder = ReparamModule(self.encoder)
        num_params = sum([np.prod(p.size()) for p in (self.encoder.parameters())])
        expert_trajectory = self.get_expert_trajectory(file)

        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm
        # features = self.data.x
        # adj_norm = self.data.edge_index
        if self.args.epochs == 0:
            with torch.no_grad():
                x = self.encoder(features, adj_norm)
            return x.detach()

        best_loss = 1e5
        pat = 0
        for i in range(self.args.epochs):
            # updated encoder params
            update_flat_params = self.encoder.flat_param
            # num_params = sum([np.prod(p.size()) for p in (net.parameters())])
            # expert_trajectory = self.get_expert_trajectory()
            start_epoch = np.random.randint(0, 10)
            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch + 360]
            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)

            # match_params = [
            #     torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            if i == 0:
                match_params = [
                    torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            else:
                match_params = [update_flat_params]

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)
            loss_dict1 = {}
            weight_dict = {}
            entropy_dict = {}
            param_loss_list = []
            param_dist_list = []

            # encoder matching
            for _ in range(380):
                forward_params = match_params[-1]
                x = self.encoder(features, adj_norm, flat_param=forward_params)
                task_loss, loss_dict, entropy_loss = self.get_task_loss(x, forward_params)
                grad = torch.autograd.grad(task_loss, match_params[-1], create_graph=True)[0]
                match_params.append(match_params[-1] - lr * grad)

            # forward_params = match_params[0]
            update_flat_params = match_params[-1]
            x = self.encoder(features, adj_norm, flat_param=update_flat_params)
            task_loss, loss_dict, entropy_loss = self.get_task_loss(x, update_flat_params)
            weight_task_loss = torch.mul(self.weight, task_loss)
            loss = torch.sum(weight_task_loss)
            for idx, (k, v) in enumerate(loss_dict.items()):
                loss_dict1[k + '_loss'] = v
            # for idx, (k, v) in enumerate(entropy_loss.items()):
            #     entropy_dict[k + '_entropy_loss'] = v
            log_dict = {k: v.item() for k, v in loss_dict1.items()}
            # log_dict.update({k: v.item() for k, v in entropy_dict.items()})
            # log_dict.update({k: v.item() for k, v in weight_loss_dict.items()})
            # log_dict.update({k: v.item() for k, v in weight_dict.items()})
            log_dict.update({"epoch": i})
            # log_dict.update({'total_loss': loss})
            # wandb.log(log_dict)
            if i == 0:
                initial_task_loss = task_loss.data.cpu()
                initial_task_loss = initial_task_loss.numpy()

            if i % 50 == 0 and verbose:
                print(f'Epoch {i}: {loss.item()}')
            if loss < best_loss:
                best_loss = loss
                best_weights = deepcopy(self.encoder.state_dict())
                pat = 0
            else:
                pat += 1
            if pat == patience:
                print('Early Stopped at Epoch %s' % i)
                break
            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)
            param_loss += torch.nn.functional.mse_loss(match_params[-1], target_params, reduction='sum')
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction='sum')

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            matching_loss = param_loss

            total_loss = task_loss + matching_loss
            log_dict.update({"matching_loss": matching_loss})
            # log_dict.update({'task_loss + grand_loss': total_loss})
            wandb.log(log_dict)

            self.optimizer.zero_grad()
            optimizer_lr.zero_grad()

            # task_loss.backward()
            # matching_loss.backward()
            total_loss.backward()

            self.optimizer.step()
            optimizer_lr.step()

            # schedular_lr.step()

            '''
            print('')
            wait = input("PRESS ENTER TO CONTINUE.")
            print('')
            '''
        # renormalize
        # normalize_coeff = len(self.ssl_agent) / torch.sum(self.weight.data, dim=0)
        # self.weight.data = self.weight.data * normalize_coeff

        self.encoder.eval()
        self.encoder.load_state_dict(best_weights)
        with torch.no_grad():
            x = self.encoder(features, adj_norm, flat_param=forward_params)
        return x.detach()

    def test(self, patience=200, file=None, verbose=True):
        expert_trajectory = self.get_expert_trajectory(file)

        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm

        if self.args.epochs == 0:
            with torch.no_grad():
                x = self.encoder(features, adj_norm)
            return x.detach()

        for i in range(self.args.epochs):
            print(f'Epoch:{i} starting')

            fix_params = expert_trajectory[i]
            initial_params = torch.cat([p.data.to(self.device).reshape(-1) for p in fix_params], 0).requires_grad_(True)

            if self.set_of_ssl[-1] == 'DGI' or self.set_of_ssl[-1] == 'DGISample':
                self.ssl_agent[-1].gcn.flat_param.data.copy_(initial_params)
                self.encoder.flat_param.data.copy_(initial_params)
            else:
                self.encoder.flat_param.data.copy_(initial_params)

            for step in range(100):
                loss_dict1 = {}

                x = self.encoder(features, adj_norm, flat_param=initial_params)
                task_loss, loss_dict, entropy_loss, acc = self.get_task_loss(x, initial_params)

                self.optimizer.zero_grad()
                task_loss.backward()
                self.optimizer.step()

                if i % 100 == 0 and i > 20:
                    for idx, (k, v) in enumerate(loss_dict.items()):
                        loss_dict1[f'task_loss_{i:04d}'] = v
                    log_dict = {k: v.item() for k, v in loss_dict1.items()}
                    log_dict.update({"step": step})
                    log_dict.update({f'acc_{i:04d}': acc})
                    wandb.log(log_dict)
                elif i < 20:
                    for idx, (k, v) in enumerate(loss_dict.items()):
                        loss_dict1[f'task_loss_{i:04d}'] = v
                    log_dict = {k: v.item() for k, v in loss_dict1.items()}
                    log_dict.update({"step": step})
                    log_dict.update({f'acc_{i:04d}': acc})
                    wandb.log(log_dict)

    def test1(self, patience=200, file=None, verbose=True):
        expert_trajectory = self.get_expert_trajectory(file)

        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm
        # features = self.data.x
        # adj_norm = self.data.edge_index
        if self.args.epochs == 0:
            with torch.no_grad():
                x = self.encoder(features, adj_norm)
            return x.detach()

        best_loss = 1e5
        pat = 0
        best_params = None
        for i in range(self.args.epochs):
            # updated encoder params
            update_flat_params = self.encoder.flat_param

            fix_params = expert_trajectory[i]
            initial_params = torch.cat([p.data.to(self.device).reshape(-1) for p in fix_params], 0).requires_grad_(True)

            loss_dict1 = {}
            if i == 0:
                forward_params = initial_params
            else:
                forward_params = update_flat_params

            distance = torch.nn.functional.mse_loss(initial_params, forward_params).detach().cpu()

            x = self.encoder(features, adj_norm, flat_param=forward_params)
            task_loss, loss_dict, entropy_loss = self.get_task_loss(x, forward_params)

            for idx, (k, v) in enumerate(loss_dict.items()):
                loss_dict1[k + '_loss'] = v
            log_dict = {k: v.item() for k, v in loss_dict1.items()}
            log_dict.update({"params_distance": distance})
            log_dict.update({"epoch": i})

            if i == 0:
                initial_task_loss = task_loss.data.cpu()
                initial_task_loss = initial_task_loss.numpy()

            if i % 50 == 0 and verbose:
                print(f'Epoch {i}: {task_loss.item()}')
            if task_loss < best_loss:
                best_loss = task_loss
                best_weights = deepcopy(self.encoder.state_dict())
                best_params = forward_params
                pat = 0
            else:
                pat += 1
            if pat == patience:
                print('Early Stopped at Epoch %s' % i)
                break

            wandb.log(log_dict)

            self.optimizer.zero_grad()
            task_loss.backward()

            self.optimizer.step()

        self.encoder.eval()
        self.encoder.load_state_dict(best_weights)
        with torch.no_grad():
            x = self.encoder(features, adj_norm, flat_param=best_params)
        return x.detach()

    def test2(self, patience=200, file=None, verbose=True):
        expert_trajectory = self.get_expert_trajectory(file)

        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm

        if self.args.epochs == 0:
            with torch.no_grad():
                x = self.encoder(features, adj_norm)
            return x.detach()

        for i in range(self.args.epochs):
            print(f'Epoch:{i} starting')
            fix_params = expert_trajectory[i]
            initial_params = torch.cat([p.data.to(self.device).reshape(-1) for p in fix_params], 0).requires_grad_(True)

            for step in range(100):
                loss_dict1 = {}
                update_flat_params = self.encoder.flat_param
                if step == 0:
                    forward_params = initial_params
                    if self.set_of_ssl[-1] == 'DGI' or self.set_of_ssl[-1] == 'DGISample':
                        self.ssl_agent[-1].gcn.flat_param.data.copy_(initial_params)
                        self.encoder.flat_param.data.copy_(initial_params)
                    else:
                        self.encoder.flat_param.data.copy_(initial_params)
                else:
                    forward_params = update_flat_params

                x = self.encoder(features, adj_norm, flat_param=forward_params)
                task_loss, loss_dict, entropy_loss, acc = self.get_task_loss(x, forward_params)

                distance = torch.nn.functional.mse_loss(initial_params, forward_params).detach().cpu() * 100

                self.optimizer.zero_grad()
                task_loss.backward()
                self.optimizer.step()

                if i % 100 == 0 and i > 20:
                    for idx, (k, v) in enumerate(loss_dict.items()):
                        loss_dict1[f'task_loss_{i:04d}'] = v
                    log_dict = {k: v.item() for k, v in loss_dict1.items()}
                    log_dict.update({f'params_distance_{i:04d}': distance})
                    log_dict.update({"step": step})
                    log_dict.update({f'acc_{i:04d}': acc})
                    wandb.log(log_dict)
                elif i < 20:
                    for idx, (k, v) in enumerate(loss_dict.items()):
                        loss_dict1[f'task_loss_{i:04d}'] = v
                    log_dict = {k: v.item() for k, v in loss_dict1.items()}
                    log_dict.update({f'params_distance_{i:04d}': distance})
                    log_dict.update({"step": step})
                    log_dict.update({f'acc_{i:04d}': acc})
                    wandb.log(log_dict)

    def pinjie(self, encoder1=None, encoder2=None, encoder3=None):
        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm
        if encoder1 and encoder2 and encoder3 is not None:
            x1 = encoder1(features, adj_norm)
            x2 = encoder2(features, adj_norm)
            x3 = encoder3(features, adj_norm)
            embedding1 = x1.detach()
            embedding2 = x2.detach()
            embedding3 = x3.detach()
            embedding = torch.cat((embedding1, embedding2, embedding3), 1)
        elif encoder1 and encoder2 is not None and encoder3 is None:
            x1 = encoder1(features, adj_norm)
            x2 = encoder2(features, adj_norm)
            embedding1 = x1.detach()
            embedding2 = x2.detach()
            embedding = torch.cat((embedding1, embedding2), 1)
        else:
            x = encoder1(features, adj_norm)
            embedding = x.detach()
        return embedding

    def get_last_shared_layer(self):
        outs = []
        for idx, ssl in enumerate(self.ssl_agent):
            outs.append(ssl.disc)
        return outs

    def get_expert_trajectory(self, file):
        buffer = []
        buffer = torch.load(os.path.join('./trajectory/photo/gcn', file))
        return buffer

    def get_task_loss(self, x, params=None):
        task_loss = []
        entropy_loss = {}
        loss_dict = {}
        acc = 0
        for ix, ssl in enumerate(self.ssl_agent):
            # loss = loss + self.weight[ix] * ssl.make_loss(x)
            if params is None:
                loss, loss2, acc = ssl.make_loss(x)
            else:
                if self.set_of_ssl[ix] is ("DGISample" or "DGI"):
                    loss, loss2, acc = ssl.make_loss(x, params)
                else:
                    loss, loss2, acc = ssl.make_loss(x)
            task_loss.append(loss)
            loss_dict[self.set_of_ssl[ix]] = loss
            entropy_loss[self.set_of_ssl[ix]] = loss2
        task_loss = torch.stack(task_loss)
        return task_loss, loss_dict, entropy_loss, acc

    def evaluate_pretrained(self, x):
        args = self.args
        idx_train = self.data.idx_train
        idx_val = self.data.idx_val
        idx_test = self.data.idx_test
        labels = self.processed_data.labels
        xent = nn.CrossEntropyLoss()

        runs = 1 if args.dataset != 'wiki' else 20
        accs = []
        for i in range(runs):

            if args.dataset == 'wiki':
                split_id = _
                idx_train = self.data.splits['train'][split_id]
                idx_val = self.data.splits['val'][split_id]
                idx_test = self.data.splits['test'][split_id]

            log = LogReg(x.shape[1], labels.max().item()+1).to(self.device)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0)

            if args.dataset in ['photo', 'computers', 'corafull']:
                epochs = 3000
            else:
                epochs = 300

            if args.dataset in ['citeseer']:
                opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.01)

            # best_val_acc = 0
            # best_weight = None
            for _ in range(epochs):
                log.train()
                opt.zero_grad()
                output = log(x[idx_train])
                loss = xent(output, labels[idx_train])
                loss.backward()
                opt.step()

                # log.eval()
                # output = log(x[idx_val])
                # val_preds = torch.argmax(output, dim=1)
                # val_acc = torch.sum(val_preds == labels[idx_val]).float() / labels[idx_val].shape[0]
                #
                # if val_acc > best_val_acc:
                #     best_val_acc = val_acc
                #     best_weight = deepcopy(log.state_dict())
                #     # print(f'best_val_acc:{best_val_acc}')

                # log.load_state_dict(best_weight)
                log.eval()
                logits = log(x[idx_test])
                test_preds = torch.argmax(logits, dim=1)
                acc = torch.sum(test_preds == labels[idx_test]).float() / labels[idx_test].shape[0]
                accs.append(acc.item() * 100)
                wandb.log({'acc': acc.item() * 100, "epoch": i})
            wandb.finish()
        print('Average accuracy:', np.mean(accs), np.std(accs))
        return np.mean(accs), np.std(accs)

        # kmeans_input = x.cpu().numpy()
        # nclass = self.data.labels.max().item()+1
        # kmeans = KMeans(n_clusters=nclass, random_state=0).fit(kmeans_input)
        # pred = kmeans.predict(kmeans_input)
        # labels = self.data.labels
        # nmi = v_measure_score(labels, pred)
        # print('Node clustering:', nmi)
        # edge_index = self.data.adj.nonzero()
        # homo = (pred[edge_index[0]] == pred[edge_index[1]])
        # print('Homo with knowledge of cluster num:', np.mean(homo))
        # return np.mean(accs), np.std(accs), nmi

    def fit_link_predictor(self, x):
        train_pos_edge_index = self.processed_data.train_pos_edge_index
        train_neg_edge_index = negative_sampling(
            edge_index=train_pos_edge_index, num_nodes=self.nums_node,
            num_neg_samples=train_pos_edge_index.size(1),
            force_undirected=True,
        )
        train_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=-1)
        train_link_labels = torch.cat(
            [torch.ones(train_pos_edge_index.shape[1]), torch.zeros(train_pos_edge_index.shape[1])]).to(self.device)
        val_pos_edge_index = self.processed_data.val_pos_edge_index
        val_neg_edge_index = self.processed_data.val_neg_edge_index
        val_edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim=-1)
        val_link_labels = torch.cat(
            [torch.ones(val_pos_edge_index.shape[1]), torch.zeros(val_pos_edge_index.shape[1])]).to(self.device)
        test_pos_edge_index = self.processed_data.test_pos_edge_index
        test_neg_edge_index = self.processed_data.test_neg_edge_index
        test_edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=-1)
        test_link_labels = torch.cat(
            [torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_pos_edge_index.shape[1])]).to(self.device)

        model = Link_Pred(x.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
        x = x.to(self.device)
        aucs = []
        for i in range(1):
            best_val_auc = 0.0
            cnt_wait = 0
            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                z = torch.mul(x[train_edge_index[0]], x[train_edge_index[1]])
                logits = model(z)
                loss = F.binary_cross_entropy_with_logits(logits, train_link_labels)
                loss.backward()
                optimizer.step()
            model.eval()
            z = torch.mul(x[test_edge_index[0]], x[test_edge_index[1]])
            test_logit = model(z)
            test_pred = test_logit.sigmoid()
            auc = roc_auc_score(test_link_labels.detach().cpu(), test_pred.detach().cpu())
            aucs.append(auc * 100)
            # wandb.log({'auc': auc * 100, "epoch": i})

        #     print(f'Epoch {i + 1}, AUC: {auc * 100}')
        # print(f'Average AUC:{np.mean(aucs)} {np.std(aucs)}')
        return np.mean(aucs), np.std(aucs)


def reset_mlp(m):
    nn.init.xavier_uniform_(m.weight.data)
    if m.bias is not None:
        m.bias.data.fill_(0.0)


class LogReg(nn.Module):
    def __init__(self, nfeat, nclass):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(nfeat, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret



class Link_Pred(torch.nn.Module):
    def __init__(self, in_channel):
        super(Link_Pred, self).__init__()
        self.linear = torch.nn.Linear(in_channel, 1)

    def forward(self, z):
        return self.linear(z).squeeze()
