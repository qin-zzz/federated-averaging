# data_utils.py
import json
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

from .config import *

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(max(0, ALL_LETTERS.find(c))) # added max to account for -1
    return indices

def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [word_to_indices(c) for c in raw_y_batch]  # to indices
    # y_batch = [letter_to_vec(c) for c in raw_y_batch]  # to one-hot
    return y_batch

def get_shakespeare(file_path, dir_path='data/shakespeare', num_client=100):

    inputs_lst = []
    targets_lst = []
    clients_lst = []
    data = {}

    with open(os.path.join(dir_path, file_path), 'r') as inf:
        cdata = json.load(inf)
    data.update(cdata['user_data'])
    list_keys = list(data.keys())

    # num_keys = len(list_keys) // percentage

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

def load_data(dataset, save_path='./data'):
    match dataset:
        case 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            trainset = datasets.MNIST(root=save_path, train=True, download=True, transform=transform)
            testset = datasets.MNIST(root=save_path, train=False, download=True, transform=transform)
            print('Dataset MNIST Loaded!')
            return trainset, testset

        case 'CIFAR10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])     
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  
            print('Dataset CIFAR10 Loaded!')
            return trainset, testset
                        
        case 'SHAKESPEARE':
            train_input, train_target, clientset = get_shakespeare('all_data_niid_2_keep_0_train_9.json')
            test_input, test_target, _ = get_shakespeare('all_data_niid_2_keep_0_test_9.json')
            trainset = TensorDataset(torch.tensor(np.asarray(train_input)), torch.tensor(np.asarray(train_target)))
            testset = TensorDataset(torch.tensor(np.asarray(test_input)), torch.tensor(np.asarray(test_target)))
            print('Dataset SHAKESPEARE Loaded!')
            return trainset, testset, clientset
    

