import os
import pickle
import random

import networkx as nx
import numpy as np
import torch
import utils
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

random.seed(42)


def onehot_encoding_node(m, embedding_dim):
    H = []
    for i in m.nodes:
        H.append(utils.node_feature(m, i, embedding_dim))
    H = np.array(H)
    return H


class BaseDataset(Dataset):
    def __init__(self, keys, data_dir, embedding_dim=20):
        self.keys = keys
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # idx = 0
        key = self.keys[idx]
        with open(os.path.join(self.data_dir, key), "rb") as f:
            data = pickle.load(f)
            if len(data) == 3:
                m1, m2, mapping = data
            else:
                m1, m2 = data
                mapping = []

        # Prepare subgraph
        n1 = m1.number_of_nodes()
        adj1 = nx.to_numpy_array(m1) + np.eye(n1)
        H1 = onehot_encoding_node(m1, self.embedding_dim)

        # Prepare source graph
        n2 = m2.number_of_nodes()
        adj2 = nx.to_numpy_array(m2) + np.eye(n2)
        H2 = onehot_encoding_node(m2, self.embedding_dim)

        # Aggregation node encoding
        agg_adj1 = np.zeros((n1 + n2, n1 + n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(H1, H2)
        dm_new = np.zeros_like(dm)
        dm_new[dm == 0.0] = 1.0
        agg_adj2[:n1, n1:] = np.copy(dm_new)
        agg_adj2[n1:, :n1] = np.copy(np.transpose(dm_new))

        H1 = np.concatenate([H1, np.zeros((n1, self.embedding_dim))], 1)
        H2 = np.concatenate([np.zeros((n2, self.embedding_dim)), H2], 1)
        H = np.concatenate([H1, H2], 0)

        # node indice for aggregation
        valid = np.zeros((n1 + n2,))
        valid[:n1] = 1

        # create mapping matrix
        mapping_matrix = np.zeros_like(agg_adj1)
        if len(mapping) > 0:
            mapping = np.array(mapping).T
            mapping[1] = mapping[1] + n1
            mapping_matrix[mapping[0], mapping[1]] = 1.0
            mapping_matrix[mapping[1], mapping[0]] = 1.0

        same_label_matrix = np.zeros_like(agg_adj1)
        same_label_matrix[:n1, n1:] = np.copy(dm_new)
        same_label_matrix[n1:, :n1] = np.copy(np.transpose(dm_new))

        # iso to class
        Y = 1 if "iso" in key else 0

        # if n1+n2 > 300 : return None
        sample = {
            "H": H,
            "A1": agg_adj1,
            "A2": agg_adj2,
            "Y": Y,
            "V": valid,
            "key": key,
            "mapping": mapping_matrix,
            "same_label": same_label_matrix,
        }

        return sample


class UnderSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        retval = np.random.choice(
            len(self.weights),
            self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    max_natoms = max([len(item["H"]) for item in batch if item is not None])

    H = np.zeros((len(batch), max_natoms, batch[0]["H"].shape[-1]))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    M = np.zeros((len(batch), max_natoms, max_natoms))
    S = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), max_natoms))

    keys = []

    for i in range(len(batch)):
        natom = len(batch[i]["H"])

        H[i, :natom] = batch[i]["H"]
        A1[i, :natom, :natom] = batch[i]["A1"]
        A2[i, :natom, :natom] = batch[i]["A2"]
        M[i, :natom, :natom] = batch[i]["mapping"]
        S[i, :natom, :natom] = batch[i]["same_label"]
        Y[i] = batch[i]["Y"]
        V[i, :natom] = batch[i]["V"]
        keys.append(batch[i]["key"])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    M = torch.from_numpy(M).float()
    S = torch.from_numpy(S).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()

    return H, A1, A2, M, S, Y, V, keys
