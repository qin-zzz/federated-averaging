import numpy as np
import random

import torch
from torch.utils.data import Subset
import torchvision

from torchvision.datasets import VisionDataset
import os.path
import json
from language_utils import process_x, process_y


def iid_partition(
    train_idcs, train_labels, m_per_shard=300, n_shards_per_client=2
):
    # split into n_shards of size m_per_shard
    m = len(train_labels)
    assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    return np.reshape(train_idcs[shards_idx], (100, m_per_shard * n_shards_per_client))

def noniid_partition(
    train_idcs, train_labels, m_per_shard=300, n_shards_per_client=2
):
    """ semi-pathological client sample partition
    1. sort examples by label, form shards of size 300 by grouping points
       successively
    2. each client is 2 random shards
    most clients will have 2 digits, at most 4
    """

    # sort
    idx = np.argsort(train_labels)
    train_idcs = train_idcs[idx]
    train_labels = train_labels[idx]

    # split into n_shards of size m_per_shard
    m = len(train_labels)
    assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    return np.reshape(train_idcs[shards_idx], (100, m_per_shard * n_shards_per_client))

def noniid_partition_dirichlet(train_idcs, train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    """
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  
    return client_idcs

class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices=None, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   


def get_shakespeare(json_path, num_client=100):
    inputs_lst = []
    targets_lst = []
    clients_lst = []
    data = {}

    with open(json_path, 'r') as inf:
        cdata = json.load(inf)
    data.update(cdata['user_data'])
    list_keys = list(data.keys())

    for (i, key) in enumerate(list_keys[:num_client]):
        # note: each time we append a list
        inputs = data[key]["x"]
        targets = data[key]["y"]

        for input_ in inputs:
            input_ = process_x(input_)
            inputs_lst.append(input_.reshape(-1))

        for target in targets:
            target = process_y(target)
            targets_lst += target[0]

        for _ in range(0, len(inputs)):
            clients_lst.append(i)

    return inputs_lst, targets_lst, clients_lst
      