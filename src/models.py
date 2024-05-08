# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import *

# 2NN
class MLPMnist(nn.Module):
    def __init__(self):
        super(MLPMnist, self).__init__()
        self.fc1 = nn.Linear(784, 200);
        self.fc2 = nn.Linear(200, 200);
        self.out = nn.Linear(200, 10);

    def forward(self, x):
        x = x.flatten(1) # [B x 784]
        x = F.relu(self.fc1(x)) # [B x 200]
        x = F.relu(self.fc2(x)) # [B x 200]
        x = self.out(x) # [B x 10]
        return x

# CNN
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, out_activation=False):
        conv1_ = self.conv1(x)
        conv2_ = self.conv2(conv1_)
        fc_ = conv2_.view(-1, 32*7*7)
        fc1_ = self.fc1(fc_).clamp(min=0)  # Achieve relu using clamp
        output = self.fc2(fc1_)
        if out_activation:
            return output, conv1_, conv2_
        else:
            return output

class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(8 * 8 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, out_activation=False):
        conv1_ = self.conv1(x)
        conv2_ = self.conv2(conv1_)
        fc_ = conv2_.view(-1, 32*8*8)
        fc1_ = self.fc1(fc_).clamp(min=0)  # Achieve relu using clamp
        output = self.fc2(fc1_)
        if out_activation:
            return output, conv1_, conv2_
        else:
            return output

class LSTMShakespeare(nn.Module):
    def __init__(self):
        super(LSTMShakespeare, self).__init__()
        self.embedding_len = 8
        self.seq_len = 80
        self.num_classes = 80
        self.n_hidden = 256
        self.batch_size = 32

        self.embeds = nn.Embedding(self.seq_len, self.embedding_len)
        self.multi_lstm = nn.LSTM(input_size=self.embedding_len, hidden_size=self.n_hidden, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, x, out_activation=False):
        x = x.to(torch.int64)
        x_ = self.embeds(x)
        h0 = torch.rand(2, x_.size(0), self.n_hidden).to(device)
        c0 = torch.rand(2, x_.size(0), self.n_hidden).to(device)
        activation, (h_n, c_n) = self.multi_lstm(x_,(h0,c0))

        fc_ = activation[:, -1, :]

        output = self.fc(fc_)
        if out_activation:
            return output, activation
        else:
            return output
        