import os, json, torch
import pickle

import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA


def create_affinity_graph1(affinity, dataset, pid, did):
    # load davis dataset,affinity(68, 442),kiba  dataset affinity(2111, 229)
    train_Y = affinity[did, pid]
    adj = np.zeros_like(affinity)
    adj[did, pid] = train_Y
    affinity_graph = getAffinityGraph(dataset, adj)
    return affinity_graph


def create_affinity_graph(affinity, dataset):
    # load davis dataset,affinity(68, 442)

    rows, cols = np.where(np.isnan(affinity) == False)  # 30056
    train_Y = affinity[rows, cols]
    adj = np.zeros_like(affinity)
    adj[rows, cols] = train_Y
    affinity_graph = getAffinityGraph(dataset, adj)
    drug_test_map, drug_test_map_weight_norm, target_test_map, target_test_map_weight_norm = None, None, None, None
    return affinity_graph, drug_test_map, drug_test_map_weight_norm, target_test_map, target_test_map_weight_norm


def affinity_graph_cold(affinity, dataset, pid, did,drug_sim_k, target_sim_k, cold):
    drug_test_map, drug_test_map_weight_norm, protein_test_map, protein_test_map_weight_norm = None, None, None, None
    # drug
    if cold == 'drug':
        train_Y = affinity[did, pid]
        adj = np.zeros_like(affinity)
        adj[did, pid] = train_Y
        train_did = np.unique(did)
        adj = adj[train_did, :]
        affinity_graph = getAffinityGraph(dataset, adj)
        drug_count = affinity.shape[0]
        test_did = np.setdiff1d(np.arange(drug_count), train_did)
        affinity_graph.train_did_index_dict = {val.item(): idx for idx, val in enumerate(train_did)}
        affinity_graph.test_did_index_dict = {val.item(): idx for idx, val in enumerate(test_did)}
        # affinity_graph.pid_index_dict = {val.item(): idx for idx, val in enumerate(np.unique(pid))}
        # affinity_graph.pid_index_dict = {val.item(): idx for idx, val in enumerate(np.unique(pid))}
        drug_sim = np.loadtxt(f"data/{dataset}/drug_drug_sim.txt", delimiter=",")
        drug_test_train_sim = drug_sim[test_did, :]
        drug_test_train_sim[:, test_did] = -1
        drug_train_count = len(train_did)
        drug_test_train_map = np.argpartition(drug_test_train_sim, -2, 1)[:,-drug_sim_k:]
        drug_train_map = np.full(drug_count, -1)
        drug_train_map[train_did] = np.arange(drug_train_count)
        drug_test_map = drug_train_map[drug_test_train_map]
        drug_test_map_weight = drug_test_train_sim[np.tile(np.expand_dims(np.arange(drug_test_train_sim.shape[0]), 0), (drug_sim_k, 1)).transpose(), drug_test_train_map]
        drug_test_map_weight_sum = np.expand_dims(np.sum(drug_test_map_weight, axis=1), axis=1)
        drug_test_map_weight_norm = np.expand_dims(drug_test_map_weight / drug_test_map_weight_sum, axis=2)
    elif cold == 'protein':
        # load davis dataset,affinity(68, 442),kiba  dataset affinity(2111, 229)
        train_Y = affinity[did, pid]
        adj = np.zeros_like(affinity)
        adj[did, pid] = train_Y
        train_pid = np.unique(pid)
        adj = adj[:, train_pid]
        affinity_graph = getAffinityGraph(dataset, adj)
        protein_count = affinity.shape[1]
        test_pid = np.setdiff1d(np.arange(protein_count), train_pid)
        affinity_graph.train_pid_index_dict = {val.item(): idx for idx, val in enumerate(train_pid)}
        affinity_graph.test_pid_index_dict = {val.item(): idx for idx, val in enumerate(test_pid)}
        protein_sim = np.loadtxt(f"data/{dataset}/target-target-sim.txt", delimiter=",")
        protein_test_train_sim = protein_sim[test_pid, :]
        protein_test_train_sim[:, test_pid] = -1
        protein_train_count = len(train_pid)
        protein_test_train_map = np.argpartition(protein_test_train_sim, -2, 1)[:,-target_sim_k:]
        protein_train_map = np.full(protein_count, -1)
        protein_train_map[train_pid] = np.arange(protein_train_count)
        protein_test_map = protein_train_map[protein_test_train_map]
        protein_test_map_weight = protein_test_train_sim[np.tile(np.expand_dims(np.arange(protein_test_train_sim.shape[0]), 0), (target_sim_k, 1)).transpose(), protein_test_train_map]
        protein_test_map_weight_sum = np.expand_dims(np.sum(protein_test_map_weight, axis=1), axis=1)
        protein_test_map_weight_norm = np.expand_dims(protein_test_map_weight / protein_test_map_weight_sum, axis=2)
    else:
        # load davis dataset,affinity(68, 442),kiba  dataset affinity(2111, 229)
        train_Y = affinity[did, pid]
        adj = np.zeros_like(affinity)
        adj[did, pid] = train_Y
        train_did = np.unique(did)
        train_pid = np.unique(pid)
        adj = adj[train_did, :]
        adj = adj[:, train_pid]
        affinity_graph = getAffinityGraph(dataset, adj)
        drug_count = affinity.shape[0]
        protein_count = affinity.shape[1]
        test_did = np.setdiff1d(np.arange(drug_count), train_did)
        test_pid = np.setdiff1d(np.arange(protein_count), train_pid)
        affinity_graph.drug_train_did_index_dict = {val.item(): idx for idx, val in enumerate(train_did)}
        affinity_graph.drug_test_did_index_dict = {val.item(): idx for idx, val in enumerate(test_did)}
        affinity_graph.protein_train_pid_index_dict = {val.item(): idx for idx, val in enumerate(train_pid)}
        affinity_graph.protein_test_pid_index_dict = {val.item(): idx for idx, val in enumerate(test_pid)}
        # drug map
        drug_sim = np.loadtxt(f"data/{dataset}/drug-drug-sim.txt", delimiter=",")
        drug_test_train_sim = drug_sim[test_did, :]
        drug_test_train_sim[:, test_did] = -1
        drug_train_count = len(train_did)
        drug_test_train_map = np.argpartition(drug_test_train_sim, -2, 1)[:, -drug_sim_k:]
        drug_train_map = np.full(drug_count, -1)
        drug_train_map[train_did] = np.arange(drug_train_count)
        drug_test_map = drug_train_map[drug_test_train_map]
        drug_test_map_weight = drug_test_train_sim[np.tile(np.expand_dims(np.arange(drug_test_train_sim.shape[0]), 0),
                                                           (drug_sim_k, 1)).transpose(), drug_test_train_map]
        drug_test_map_weight_sum = np.expand_dims(np.sum(drug_test_map_weight, axis=1), axis=1)
        drug_test_map_weight_norm = np.expand_dims(drug_test_map_weight / drug_test_map_weight_sum, axis=2)
        # protein map
        protein_sim = np.loadtxt(f"data/{dataset}/target-target-sim.txt", delimiter=",")
        protein_test_train_sim = protein_sim[test_pid, :]
        protein_test_train_sim[:, test_pid] = -1
        protein_train_count = len(train_pid)
        protein_test_train_map = np.argpartition(protein_test_train_sim, -2, 1)[:, -target_sim_k:]
        protein_train_map = np.full(protein_count, -1)
        protein_train_map[train_pid] = np.arange(protein_train_count)
        protein_test_map = protein_train_map[protein_test_train_map]
        protein_test_map_weight = protein_test_train_sim[
            np.tile(np.expand_dims(np.arange(protein_test_train_sim.shape[0]), 0),
                    (target_sim_k, 1)).transpose(), protein_test_train_map]
        protein_test_map_weight_sum = np.expand_dims(np.sum(protein_test_map_weight, axis=1), axis=1)
        protein_test_map_weight_norm = np.expand_dims(protein_test_map_weight / protein_test_map_weight_sum, axis=2)
    return affinity_graph, drug_test_map, drug_test_map_weight_norm, protein_test_map, protein_test_map_weight_norm


class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]),
                            y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def getAffinityGraph(dataset, adj, weighted=True, drug_aff_k=40, target_aff_k=150):
    num_drugs = adj.shape[0]
    num_targets = adj.shape[1]
    if dataset == "davis":
        adj[adj <= 6] = 0
        adj[adj != 0] -= 6
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == 'kiba':
        # adj[adj <= 11] = 0
        # adj[adj != 0] -= 10
        adj[adj <= 12.1] = 0
        adj[adj != 0] -= 12.1
        # adj[adj <= 13] = 0
        # adj[adj != 0] -= 13
        adj_norm = minMaxNormalize(adj, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T

    adj = np.concatenate((
        np.concatenate((np.zeros([num_drugs, num_drugs]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_targets, num_targets])), 1)
    ), 0)

    train_raw_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_raw_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj[train_raw_ids, train_col_ids]

    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drugs, 1)),
        np.tile(np.array([0, 1]), (num_targets, 1))
    ), 0)

    adj_features = np.zeros_like(adj)
    adj_features[adj != 0] = 1

    features = np.concatenate((node_type_features, adj_features), 1)  # [68+442,2+510]

    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs)) if weighted \
        else DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj_features),
                       edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_node1s", num_drugs)
    affinity_graph.__setitem__("num_node2s", num_targets)

    return affinity_graph


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj


def read_data(dataset):
    dataset_path = 'data/' + dataset + '/'
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    return affinity


if __name__ == "__main__":
    dataset = 'kiba'
    # dataset = 'davis'
    affinity = read_data(dataset)
    affinity_graph = create_affinity_graph(affinity, dataset)
