import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_f

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, n_signals, window_length, n_assets, seed, fc1, fc2):
        """Initialize parameters and build model.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            n_assets (int): Number of assets in portfolio not counting cash
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """

        super(Actor, self).__init__()

        out_channels = n_signals
        kernel_size = 3
        self.n_assets = n_assets
        self.n_signals = n_signals
        self.conv1d_out = (window_length - kernel_size + 1) * out_channels
        self.seed = torch.manual_seed(seed)
        self.conv1d = [nn.Conv1d(n_signals, out_channels, kernel_size=kernel_size) for _ in range(n_assets)]
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = [nn.Linear(self.conv1d_out, fc1) for _ in range(n_assets)]
        self.fc2 = [nn.Linear(fc1, fc2) for _ in range(n_assets)]
        self.fc3 = nn.Linear(fc2 * n_assets, n_assets)
        self.reset_parameters()

    def reset_parameters(self):
        for a in range(self.n_assets):
            nn.init.xavier_uniform_(self.conv1d[a].weight)
            self.fc1[a].weight.data.uniform_(*hidden_init(self.fc1[a]))
            self.fc2[a].weight.data.uniform_(*hidden_init(self.fc2[a]))
            self.fc1[a].bias.data.fill_(0.1)
            self.fc2[a].bias.data.fill_(0.1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        x_list = []
        for a in range(self.n_assets):
            x = self.relu(self.conv1d[a](state[:, a*self.n_signals:(a+1)*self.n_signals]))
            x = x.contiguous().view(-1, self.conv1d_out)
            x = self.relu(self.fc1[a](x))
            x_list.append(self.relu(self.fc2[a](x)))
        return self.fc3(torch.cat(x_list, dim=1))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, n_signals, window_length, n_assets, seed, fc1, fc2):
        """Initialize parameters and build model.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            n_assets (int): Number of assets in portfolio not counting cash
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """

        super(Critic, self).__init__()
        out_channels = n_signals
        kernel_size = 3
        self.n_assets = n_assets
        self.n_signals = n_signals
        self.conv1d_out = (window_length - kernel_size + 1) * out_channels
        self.seed = torch.manual_seed(seed)
        self.conv1d = [nn.Conv1d(n_signals, out_channels, kernel_size=kernel_size) for _ in range(n_assets)]
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = [nn.Linear(self.conv1d_out, fc1) for _ in range(n_assets)]
        self.fc2 = [nn.Linear(fc1, fc2) for _ in range(n_assets)]
        self.fc3 = nn.Linear(fc2 * n_assets + n_assets, n_assets)
        self.reset_parameters()

    def reset_parameters(self):
        for a in range(self.n_assets):
            nn.init.xavier_uniform_(self.conv1d[a].weight)
            self.fc1[a].weight.data.uniform_(*hidden_init(self.fc1[a]))
            self.fc2[a].weight.data.uniform_(*hidden_init(self.fc2[a]))
            self.fc1[a].bias.data.fill_(0.1)
            self.fc2[a].bias.data.fill_(0.1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x_list = []
        for a in range(self.n_assets):
            x = self.relu(self.conv1d[a](state[:, a*self.n_signals:(a+1)*self.n_signals]))
            x = x.contiguous().view(-1, self.conv1d_out)
            x = self.relu(self.fc1[a](x))
            x_list.append(self.relu(self.fc2[a](x)))
        x_list.append(action)
        return self.fc3(torch.cat(x_list, dim=1))
