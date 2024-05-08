# sampling.py
import math
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from .config import *

def data_to_tensor(data):
    """ Loads dataset to memory, applies transform"""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data))
    img, label = next(iter(loader))
    return img, label

def iid_partition(data_train, bsz, num_clients):
    """
    Sample I.I.D. client data for MNIST and CIFAR10
    :param data_train:
    :param bsz:
    :param num_clients:
    :return: dataloader for each client
    """
    num_items = int(len(data_train)/num_clients)
    list_users, all_idxs = [], [i for i in range(len(data_train))]
    for i in range(num_clients):
        selected_items = set(np.random.choice(all_idxs, num_items, replace=False))
        list_users.append(sorted(selected_items))
        all_idxs = list(set(all_idxs) - selected_items)
    return [DataLoader(data_train, batch_size=bsz, sampler=SubsetRandomSampler(indices)) for indices in list_users]

# https://github.com/alexbie98/fedavg/blob/main/data.py
def mnist_noniid_partition(
    data_train, bsz, num_clients, m_per_shard, n_shards_per_client
):
    """ 
    Sample non-I.I.D client data from MNIST dataset
    1. sort examples by label, form shards of size 300 by grouping points
       successively
    2. each client is 2 random shards
    most clients will have 2 digits, at most 4
    """

    # load data into memory
    img, label = data_to_tensor(data_train)

    # sort
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    # split into n_shards of size m_per_shard
    m = len(data_train)
    # assert m % m_per_shard == 0
    assert num_clients * m_per_shard * n_shards_per_client == m
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    # pick shards to create a dataset for each client
    assert n_shards % n_shards_per_client == 0
    n_clients = n_shards // n_shards_per_client
    client_data = [
        torch.utils.data.TensorDataset(
            torch.cat([img[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)]),
            torch.cat([label[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)])
        )
        for i in range(n_clients)
    ]

    # make dataloaders
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader

# https://github.com/PengchaoHan/EasyFL/blob/main/util/sampling.py
def shakespeare_noniid_partition(data_train, bsz, num_clients, client_list):
    """
    Sample non-I.I.D client data from shakespeare dataset
    :param data_train:
    :param num_clients:
    :return:
    """
    # num_shards, num_imgs = 2*num_users, int(dataset.data.size()[0]/2/num_users)  # choose two number from a set with num_shards, each client has 2*num_imgs images
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(dataset.data.size()[0])
    # labels = dataset.train_labels.numpy()
    #
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]
    #
    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

    label_list = np.asarray(client_list)
    minLabel = min(label_list)
    numLabels = len(set(label_list))

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    for i in range(0, len(label_list)):
        tmp_target_node = int((label_list[i] - minLabel) % num_clients)
        if num_clients > numLabels:
            tmpMinIndex = 0
            tmpMinVal = math.inf
            for n in range(0, num_clients):
                if (n) % numLabels == tmp_target_node and len(dict_users[n]) < tmpMinVal:
                    tmpMinVal = len(dict_users[n])
                    tmpMinIndex = n
            tmp_target_node = tmpMinIndex
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)
    dict_users = list(map(lambda kv: kv[1].tolist(), dict_users.items()))
    return [DataLoader(data_train, batch_size=bsz, sampler=SubsetRandomSampler(indices)) for indices in dict_users]


def split_data(dataset, data_train, num_clients, iid=True, batch_size=1, client_list=None):
    match dataset:
        case 'MNIST':
            if iid:
                dict_users = iid_partition(data_train, bsz=batch_size, num_clients=num_clients)
            else:
                dict_users = mnist_noniid_partition(data_train, bsz=batch_size, num_clients=100, m_per_shard=300, n_shards_per_client=2) 
        case 'CIFAR10':
            if iid:
                dict_users = iid_partition(data_train, bsz=batch_size, num_clients=num_clients)
            else:
                raise Exception('Only consider IID setting in CIFAR10')
        case 'SHAKESPEARE':
            if iid:
                raise Exception('Only consider NON-IID setting in SHAKESPEARE')
            else:
                dict_users = shakespeare_noniid_partition(data_train, bsz=batch_size, num_clients=num_clients, client_list=client_list)
    return dict_users

