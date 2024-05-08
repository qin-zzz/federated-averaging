# helper.py
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from .config import *

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]

def display_train_stats(cfl_stats, title = "", label="", target_accuracy=0.97):
    # clear_output(wait=True)
    communication_rounds = max(cfl_stats.rounds)
    
    fig = plt.figure(figsize=(10,3))
    plt.title(title)

    plt.subplot(1,2,1)

    plt.plot(cfl_stats.rounds, cfl_stats.acc_clients, color="C0")
    
    plt.axhline(y=target_accuracy, linestyle="--", color="r")
    plt.legend()

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.subplot(1,2,2)
    
    plt.plot(cfl_stats.rounds, cfl_stats.train_loss, color="C1", label=r"Train Loss")
    plt.plot(cfl_stats.rounds, cfl_stats.test_loss, color="C2", label=r"Test Loss")

    plt.xlabel("Communication Rounds")
    plt.legend()
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0, 2)

    return fig


def plot_accuracy(dataset, keyword, num_epoch=100):
    target_accuracy = target_accuracy_set[dataset]

    stats_dir = os.path.join(stats_dir, dataset)
    names = [f for f in os.listdir(stats_dir) if re.match(keyword, f)]

    plt.figure(figsize=(8,6))

    # plt.axis([0, 100, 0.9, 1])
    x = np.arange(1,101)

    for fname in names:
        print(fname)
        l = fname.split('.')[0].split('_')
        c, b, e = l[-3], l[-2], l[-1]
        stats = np.load(os.path.join(stats_dir, fname), allow_pickle=True).item()
        plt.plot(x, stats.acc_clients, label=f'C={c}, B={b}, E={e}')
    
    plt.axhline(y=target_accuracy, linestyle="--", color="r", label=f'Target={target_accuracy}')
    plt.title(f"FedAvg test accuracy after {num_epoch} rounds on {dataset}({keyword})")
    plt.xlabel("Communication rounds $t$")
    plt.ylabel("Test accuracy")
    plt.legend()

    plt.show()
# plot_accuracy('MNIST', keyword='mlp_iid')