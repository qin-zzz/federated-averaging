# simulate.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .fl_devices import train_client, aggregate_models, validate
from .helper import ExperimentLogger
from .data_utils import load_data
from .models import MLPMnist, CNNMnist, CNNCifar10, LSTMShakespeare
from .sampling import split_data
from .config import *

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

def simulate(dataset, num_clients, num_participants, batch_size, num_local_epochs, num_rounds, learning_rate, target_accuracy, iid=True, mlp=False, saved=True):
    match dataset:
        case 'MNIST':
            if mlp:
                global_model = MLPMnist().to(device)
                local_models = [MLPMnist().to(device) for _ in range(num_clients)]
                kw = 'mlp_'
            else:             
                global_model = CNNMnist().to(device)
                local_models = [CNNMnist().to(device) for _ in range(num_clients)]
                kw = 'cnn_'
            trainset, testset = load_data(dataset)
            client_loaders = split_data(dataset, trainset, num_clients, iid=iid)
        case 'CIFAR10':
            global_model = CNNCifar10().to(device)
            local_models = [CNNCifar10().to(device) for _ in range(num_clients)]
            kw = 'cnn_'
            trainset, testset = load_data(dataset)
            client_loaders = split_data(dataset, trainset, num_clients, iid=iid)
        case 'SHAKESPEARE':
            global_model = LSTMShakespeare().to(device)
            local_models = [LSTMShakespeare().to(device) for _ in range(num_clients)]
            kw = 'lstm_'
            trainset, testset, clientset = load_data(dataset)
            client_loaders = split_data(dataset, trainset, num_clients, iid=iid, client_list=clientset)
        
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    optimizers = [optim.SGD(model.parameters(), lr=learning_rate) for model in local_models]  

    cfl_stats = ExperimentLogger()

    for round in range(num_rounds):
        selected_clients_ids = np.random.permutation(num_clients)[:num_participants]

        # client_models = [train_client(local_models[id], optimizers[id], client_loaders[id], global_model, num_local_epochs) for id in selected_clients_ids]
        client_models, train_loss = [], 0
        for id in selected_clients_ids:
            client_model, loss = train_client(local_models[id], optimizers[id], client_loaders[id], global_model, num_local_epochs)
            client_models.append(client_model)
            train_loss += loss
        
        global_model = aggregate_models(global_model, client_models)

        accuracy, test_loss = validate(global_model, test_loader)
        
        train_loss /= num_participants

        print(f"Round {round + 1} | ids: {sorted(selected_clients_ids)} | average accuracy: {accuracy:.4f} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f}")
        cfl_stats.log({"rounds" : round, "train_loss":train_loss, "test_loss":test_loss, "acc_clients": accuracy})
    
    kw += 'iid' if iid else 'noniid'

    if saved:
        np.save(f'{stats_dir_root}/{dataset}/{kw}_{num_participants}_{batch_size}_{num_local_epochs}.npy', cfl_stats) 

    return cfl_stats, kw
