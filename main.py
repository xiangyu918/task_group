import cma
from deeprobust.graph.data import Pyg2Dpr
import argparse
import numpy as np
from SSL import SSL
from selfsl.gnn_encoder import GCN
from utils import *
from sklearn.metrics import roc_auc_score
import torch
import warnings
import wandb
import itertools
import os
from reparam_module import ReparamModule

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='computers')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0000)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--fix_weight', type=int, default=1)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--mode', type=str, default='grad_norm')
# parser.add_argument('--mode', choices=('grad_norm', 'equal_weight'), default='grad_norm')
parser.add_argument('--alpha', type=float, default=0.12)
parser.add_argument('--sigma', type=float, default=100.0)
args = parser.parse_args()

import logging
LOG_FILENAME = f'logs/{args.dataset}_{args.seed}.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

torch.cuda.set_device(args.gpu_id)
print('===========')
print(args)
logging.warn(args)


np.random.seed(args.seed)
data = get_dataset(args.dataset, args.normalize_features)
nfeat = data.features.shape[1]

save_dir = os.path.join('./trajectory', 'photo')
save_dir = os.path.join(save_dir, 'gcn')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_average_params(encoder1, encoder2):
    encoder1_params = encoder1.state_dict()
    encoder2_params = encoder2.state_dict()
    # encoder3_params = encoder3.state_dict()
    average_params = {}
    for key in encoder1_params:
        average_params[key] = (encoder1_params[key] + encoder2_params[key]) / 2.0
    return average_params


def run():
    # wandb.login(key="55ebdf3ed93b245f2b03bb1933cb7973971d6ca2")
    wandb.init(config=sweep_config)
    # wandb.init(project='computers_concat')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    i = 2

    set_of_ssl = wandb.config.my_param
    # set_of_ssl = ['PairwiseDistance', 'PairwiseAttrSim']

    # filename = "replay_buffer_Par.pt"
    filename = "replay_buffer_PairwiseAttrSim_{}.pt".format(i)

    print(f'task: {set_of_ssl}')
    print(f'trajectory: {filename}')
    if args.dataset == 'arxiv':
        local_set_of_ssl = ['Clu', 'DGISample', 'PairwiseDistance', 'PairwiseAttrSim']
        encoder = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout, nlayers=args.num_layers, activation='relu',
                      with_bn=True)
    else:
        encoder = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout)
        encoder1 = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout)
        encoder2 = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout)
        encoder3 = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout)
    if data.adj.shape[0] > 5000:
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl
    # encoder = ReparamModule(encoder)
    # encoder.load_state_dict(torch.load("model/computers/PairwiseAttrSim.pth"))
    # encoder1.load_state_dict(torch.load("model/computers/PairwiseAttrSim.pth"))
    # encoder2.load_state_dict(torch.load("model/computers/PairwiseDistance.pth"))
    # encoder3.load_state_dict(torch.load("model/computers/PairwiseDistance.pth"))
    # average_params = get_average_params(encoder1, encoder2)
    # encoder.load_state_dict(average_params)
    auto_agent = SSL(data, encoder, local_set_of_ssl, save_dir, args)
    # encoder1.load_state_dict(torch.load("model/computers/DGI.pth"))
    # encoder2.load_state_dict(torch.load("model/computers/PairwiseAttrSim.pth"))
    # encoder3.load_state_dict(torch.load("model/computers/PairwiseDistance.pth"))
    # x = auto_agent.pinjie(encoder1.to('cuda'), encoder2.to('cuda'), encoder3.to('cuda'))
    x = auto_agent.pretrain(patience=3000, filename=filename, verbose=False)
    # x = auto_agent.matching(patience=50, file=filename, verbose=False)
    # auto_agent.test(patience=200, file=filename, verbose=False)
    acc, acc_std = auto_agent.evaluate_pretrained(x)
    print(f"{set_of_ssl} finished......")
    # auc, auc_std = auto_agent.fit_link_predictor(x)
    # print(f'pretrain Acc、Acc_std、AUC and AUC_std:{acc} {acc_std} {auc} {auc_std}')
    # logging.debug(f'pretrain Acc、Acc_std、AUC and AUC_std:{acc} {acc_std} {auc} {auc_std}')
    # wandb.finish()


# run()
#
# os.environ["WANDB_API_KEY"] = "55ebdf3ed93b245f2b03bb1933cb7973971d6ca2"
# os.environ["WANDB_MODE"] = "offline"
wandb.login(key="55ebdf3ed93b245f2b03bb1933cb7973971d6ca2")
task_list = ['Par', 'Clu', 'DGI', 'PairwiseDistance', 'PairwiseAttrSim']
# task_list = ['Clu', 'DGISample', 'PairwiseDistance', 'PairwiseAttrSim']
all_combinations = []
for i in range(1, 6):
    combinations = itertools.combinations(task_list, i)
    all_combinations.extend(combinations)
# print(len(all_combinations))
# print(all_combinations)

sweep_config = {
    'method': 'grid',
    'parameters': {
        'my_param': {
            'values': all_combinations
        }
    }
}
config = wandb.config
sweep_id = wandb.sweep(sweep_config, project='online_epoch_3000')
wandb.agent(sweep_id, function=run)


# wandb.login(key="584001bc7f45eea4c232828c5815c08c885725d8")
# sweep_config = {
#     'method': 'grid',
#     'parameters': {
#         'my_param': {
#             'values': [i + 1 for i in range(10)]
#         }
#     }
# }
# config = wandb.config
# sweep_id = wandb.sweep(sweep_config, project='fix_encoder')
# wandb.agent(sweep_id, function=run)

# wandb.login(key="584001bc7f45eea4c232828c5815c08c885725d8")
# sweep_config = {
#     'method': 'grid',
#     'parameters': {
#         'my_param': {
#             'values': [['Clu'], ['PairwiseAttrSim'], ['Par']]
#         }
#     }
# }
# config = wandb.config
# sweep_id = wandb.sweep(sweep_config, project='trajectory_matching1')
# wandb.agent(sweep_id, function=run)



# wandb.login(key="584001bc7f45eea4c232828c5815c08c885725d8")
# sweep_config = {
#     'method': 'grid',
#     'parameters': {
#         'my_param': {
#             'values': ["test", "test1", "test2"]
#         }
#     }
# }
# config = wandb.config
# sweep_id = wandb.sweep(sweep_config, project='trajectory_matching5')
# wandb.agent(sweep_id, function=run)
