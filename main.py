
from src.simulate import simulate
from src.helper import display_train_stats, ExperimentLogger, plot_accuracy
from src.config import *

def run(dataset, num_clients, c, b, e, n_rounds, lr, iid=True, mlp=True, saved=True):

    img_dir = f'{img_dir_root}/{dataset}'
    t_acc = target_accuracy_set[dataset]

    label_parameters = f"C={c/100: .2f}, B={b},  E={e}, ACC={t_acc}"
    print(label_parameters)
    stats, kw = simulate(dataset, num_clients, num_participants=c, batch_size=b, num_local_epochs=e, num_rounds=n_rounds, learning_rate=lr, target_accuracy=t_acc, iid=True, mlp=False, saved=True)
    name = f"{dataset}_{kw}_{c}_{b}_{e}"
    fig = display_train_stats(stats, title=name, label=label_parameters, target_accuracy=t_acc)
    fig.savefig(f'{img_dir}/{kw}_{c}_{b}_{e}_test.png')

if __name__ == '__main__':
    dataset = 'MNIST'
    num_clients = 100
    c, b, e = 10, 10, 5
    n_rounds = 1
    lr = 0.01
    run(dataset, num_clients, c, b, e, n_rounds, lr, saved=False)