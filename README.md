# Federated Averaging

This is a reproduction of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629).

The simulator is implemented in `fl_devices.py`. Experiment results on three datasets, namely MNIST, CIFAR10 and Shakespeare, are shown in `xxx.ipynb`.

All experiments consist of the following steps:
- Loading data
- Distributing Data (IID VS non-IID)
- Building Models
- Training

The major parameters in the model:
- `C`: the fraction of clients that perform computation on each round
- `B`: the local minibatch size used for the client updates
- `E`: the number of training passes each client makes over its local dataset on each round

Target results of all experiments from the paper:

| | MNIST 2NN IID | MNIST 2NN non-IID | MNIST CNN IID | MNIST CNN non-IID | CIFAR CNN IID | Shakespeare LSTM non-IID |
| -------- | -------- | ------- | -------- | ------- | -------- | ------- |
| Target Accuracy | 97% | 97% | 99% | 99% | 80% | 54% | 
| Target Rounds | <100 | <800 | <20 | <300 | <500 | <600 | 

To do:
- Experiment results on CIFAR10 and Shakespeare are by far not well
- Increasing parallelism