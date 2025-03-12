
import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Coauthor
from torch_geometric.nn import GCNConv

from model import Encoder, Model, VGAE
from collections import defaultdict
from torch_geometric.utils import remove_self_loops, add_self_loops
import function as func
import logreg as logreg
import scipy.sparse as sp
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_config(path):
    with open(path) as file:
        return yaml.load(file, Loader=SafeLoader)


def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_hidden = trial.suggest_categorical('num_hidden', [256, 512, 1024])
    tau = trial.suggest_categorical('tau', [1, 5, 10])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    num_epochs = trial.suggest_int('num_epochs', 20, 302)
    num = trial.suggest_int('num', 100, 500)
    k1 = trial.suggest_int('k1', 5, 20)
    Beta = trial.suggest_loguniform('Beta', 1e-6, 100)

    config = load_config(args.config)[args.dataset]
    torch.manual_seed(config['seed'])
    random.seed(56678)
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = Planetoid(path, args.dataset)
    nb_classes = dataset.num_classes
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    edge = data.edge_index.clone()
    adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge.cpu()[0, :], edge.cpu()[1, :])),
                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32).toarray()
    adj = torch.from_numpy(adj).cuda()

    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    if data.train_mask.dtype == torch.bool:
        idx_train = data.train_mask.nonzero(as_tuple=True)[0]

    if data.val_mask.dtype == torch.bool:
        idx_val = data.val_mask.nonzero(as_tuple=True)[0]

    if data.test_mask.dtype == torch.bool:
        idx_test = data.test_mask.nonzero(as_tuple=True)[0]

    adj_lists = defaultdict(set)
    data.edge_index, _ = add_self_loops(data.edge_index)
    for i in range(data.edge_index.size(1)):
        adj_lists[data.edge_index[0][i].item()].add(data.edge_index[1][i].item())
    edge_index2, _ = remove_self_loops(data.edge_index)

    encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model).to(device)
    vgae = VGAE(in_channels=num_hidden, hidden_channels=num_hidden, out_channels=num_hidden, activation=activation,
                device=device)
    model = Model(encoder, vgae, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0
    for epoch in range(1, num_epochs + 1):
        nodes_batch = torch.randint(0, data.num_nodes, (num,))
        node_neighbor_cen = func.sub_sam(nodes_batch, adj_lists, k1)
        model.train()
        optimizer.zero_grad()
        z1, z2 = model(data.x, data.edge_index, edge_index2)
        vgae_loss = model.vgae_loss(adj, model.vgae.mean, model.vgae.logstd)
        loss = model.loss(z1, z2, adj, node_neighbor_cen) + Beta * vgae_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            embeds = model.embed(data.x, data.edge_index).detach()

            train_embs = embeds[idx_train]
            train_lbls = data.y[idx_train]
            log = logreg.LogReg(num_hidden, nb_classes)
            log.cuda()
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.train()
            for _ in range(100):
                opt.zero_grad()
                logits = log(train_embs)
                loss = nn.CrossEntropyLoss()(logits, train_lbls)
                loss.backward()
                opt.step()

            val_embs = embeds[idx_val]
            val_lbls = data.y[idx_val]
            log.eval()
            with torch.no_grad():
                logits = log(val_embs)
                preds = torch.argmax(logits, dim=1)
                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()

    torch.save(best_model_state, args.savepath)

    return best_val_acc.item()


def train_and_evaluate(trial, dataset, device, idx_train, idx_val, idx_test, nb_classes, num_runs=5):
    learning_rate = trial.params['learning_rate']
    num_hidden = trial.params['num_hidden']
    tau = trial.params['tau']
    weight_decay = trial.params['weight_decay']
    num_epochs = trial.params['num_epochs']
    num = trial.params['num']
    k1 = trial.params['k1']
    Beta = trial.params['Beta']

    config = load_config(args.config)[args.dataset]
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]

    data = dataset[0].to(device)

    edge = data.edge_index.clone()
    adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge.cpu()[0, :], edge.cpu()[1, :])),
                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32).toarray()
    adj = torch.from_numpy(adj).cuda()

    adj_lists = defaultdict(set)
    data.edge_index, _ = add_self_loops(data.edge_index)
    for i in range(data.edge_index.size(1)):
        adj_lists[data.edge_index[0][i].item()].add(data.edge_index[1][i].item())
    edge_index2, _ = remove_self_loops(data.edge_index)

    test_accs = []
    for run in range(num_runs):
        torch.manual_seed(run)
        random.seed(run)

        encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model).to(device)
        vgae = VGAE(in_channels=num_hidden, hidden_channels=num_hidden, out_channels=num_hidden, activation=activation,
                    device=device)
        model = Model(encoder, vgae, tau).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(1, num_epochs + 1):
            nodes_batch = torch.randint(0, data.num_nodes, (num,))
            node_neighbor_cen = func.sub_sam(nodes_batch, adj_lists, k1)
            model.train()
            optimizer.zero_grad()
            z1, z2 = model(data.x, data.edge_index, edge_index2)
            vgae_loss = model.vgae_loss(adj, model.vgae.mean, model.vgae.logstd)
            loss = model.loss(z1, z2, adj, node_neighbor_cen) + Beta * vgae_loss
            loss.backward()
            optimizer.step()

        model.eval()
        embeds = model.embed(data.x, data.edge_index).detach()

        test_embs = embeds[idx_test]
        test_lbls = data.y[idx_test]
        train_embs = embeds[idx_train]
        train_lbls = data.y[idx_train]

        log = logreg.LogReg(num_hidden, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.train()
        for _ in range(100):
            opt.zero_grad()
            logits = log(train_embs)
            loss = nn.CrossEntropyLoss()(logits, train_lbls)
            loss.backward()
            opt.step()

        log.eval()
        with torch.no_grad():
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_accs.append(test_acc.item())

    return test_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--savepath', type=str, default='save/Cora.pkl')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = Planetoid(path, args.dataset)
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    if data.train_mask.dtype == torch.bool:
        idx_train = data.train_mask.nonzero(as_tuple=True)[0]

    if data.val_mask.dtype == torch.bool:
        idx_val = data.val_mask.nonzero(as_tuple=True)[0]

    if data.test_mask.dtype == torch.bool:
        idx_test = data.test_mask.nonzero(as_tuple=True)[0]

    nb_classes = dataset.num_classes

    test_accs = train_and_evaluate(trial, dataset, device, idx_train, idx_val, idx_test, nb_classes)

    print(f"Test accuracies with different seeds: {test_accs}")
    print(f"Mean test accuracy: {np.mean(test_accs)}, Std: {np.std(test_accs)}")