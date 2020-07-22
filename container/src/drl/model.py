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

        self.n_signals = n_signals

        # TODO: This need to be removed per issue #3
        self.a1 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a2 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a3 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a4 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a5 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a6 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a7 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a8 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a9 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a10 = AssetModel(n_signals, window_length, seed, fc1, fc2)

        if fc2 == 0:
            self.fc_out = nn.Linear(fc1 * n_assets, n_assets)
        else:
            self.fc_out = nn.Linear(fc2 * n_assets, n_assets)
        self.reset_parameters()

    def reset_parameters(self):

        # TODO: This need to be removed per issue #3
        self.a1.reset_parameters()
        self.a2.reset_parameters()
        self.a3.reset_parameters()
        self.a4.reset_parameters()
        self.a5.reset_parameters()
        self.a6.reset_parameters()
        self.a7.reset_parameters()
        self.a8.reset_parameters()
        self.a9.reset_parameters()
        self.a10.reset_parameters()

        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_out.bias.data.fill_(0.1)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        # TODO: This need to be removed per issue #3
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = torch.split(state, self.n_signals, 1)
        x1 = self.a1(s1)
        x2 = self.a2(s2)
        x3 = self.a3(s3)
        x4 = self.a4(s4)
        x5 = self.a5(s5)
        x6 = self.a6(s6)
        x7 = self.a7(s7)
        x8 = self.a8(s8)
        x9 = self.a9(s9)
        x10 = self.a10(s10)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), 1)

        return self.fc_out(x)


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
        self.n_signals = n_signals

        # TODO: This need to be removed per issue #3
        if n_assets != 10:
            print("ERROR:  Only operational to 10 assets per issue #3.")
            raise RuntimeError
        self.a1 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a2 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a3 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a4 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a5 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a6 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a7 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a8 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a9 = AssetModel(n_signals, window_length, seed, fc1, fc2)
        self.a10 = AssetModel(n_signals, window_length, seed, fc1, fc2)

        if fc2 == 0:
            self.fc_out = nn.Linear(fc1 * n_assets + n_assets, n_assets)
        else:
            self.fc_out = nn.Linear(fc2 * n_assets + n_assets, n_assets)
        self.reset_parameters()

    def reset_parameters(self):

        # TODO: This need to be removed per issue #3
        self.a1.reset_parameters()
        self.a2.reset_parameters()
        self.a3.reset_parameters()
        self.a4.reset_parameters()
        self.a5.reset_parameters()
        self.a6.reset_parameters()
        self.a7.reset_parameters()
        self.a8.reset_parameters()
        self.a9.reset_parameters()
        self.a10.reset_parameters()

        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_out.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        # TODO: This need to be removed per issue #3
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = torch.split(state, self.n_signals, 1)
        x1 = self.a1(s1)
        x2 = self.a2(s2)
        x3 = self.a3(s3)
        x4 = self.a4(s4)
        x5 = self.a5(s5)
        x6 = self.a6(s6)
        x7 = self.a7(s7)
        x8 = self.a8(s8)
        x9 = self.a9(s9)
        x10 = self.a10(s10)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, action), 1)

        return self.fc_out(x)


class AssetModel(nn.Module):
    def __init__(self, n_signals, window_length, seed, fc1, fc2):
        """Network built for each asset.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """
        super(AssetModel, self).__init__()
        out_channels = n_signals
        kernel_size = 3
        self.use_fc2 = fc2 > 0
        self.n_signals = n_signals
        self.conv1d_out = (window_length - kernel_size + 1) * out_channels
        self.seed = torch.manual_seed(seed)

        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv1d = nn.Conv1d(n_signals, out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)

        self.fc1 = nn.Linear(self.conv1d_out, fc1)
        if self.use_fc2:
            self.fc2 = nn.Linear(fc1, fc2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1d.weight)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0.1)
        if self.use_fc2:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0.1)

    def forward(self, state):
        # x = self.relu(self.bn(self.conv1d(state)))
        # x = self.drop1(self.relu(self.bn(self.conv1d(state))))
        # x = self.drop1(self.relu(self.conv1d(state)))
        x = self.relu(self.conv1d(state))
        x = x.contiguous().view(-1, self.conv1d_out)
        x = self.drop2(self.relu(self.fc1(x)))
        if self.use_fc2:
            x = self.drop2(self.relu(self.fc2(x)))
        return x
