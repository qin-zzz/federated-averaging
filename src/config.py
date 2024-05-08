import torch

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

target_accuracy_set = {'MNIST': 0.97,
                   'CIFAR10': 0.80,
                   'SHAKESPEARE': 0.54}

img_dir_root = f'./img'
stats_dir_root = f'./stats'