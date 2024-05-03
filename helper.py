import IPython
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]



def display_train_stats(cfl_stats, communication_rounds, title = "", target_accuracy=0.99):
    clear_output(wait=True)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    acc_mean = np.mean(cfl_stats.acc_clients, axis=1)
    acc_std = np.std(cfl_stats.acc_clients, axis=1)
    plt.fill_between(cfl_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(cfl_stats.rounds, acc_mean, color="C0")
    
    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    plt.axhline(y=target_accuracy, linestyle="--", color="k", label=r"$\varepsilon_1$")
    
    plt.title(title)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)
    
    plt.show()