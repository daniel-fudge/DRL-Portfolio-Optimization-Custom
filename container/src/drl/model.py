import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_f


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, in_channels, window_length, action_size, seed, fc1, fc2):
        """Initialize parameters and build model.

        Args:
            in_channels (int): Dimension of each state
            window_length (int): Number of days in sliding window
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """

        out_channels = in_channels
        kernel_size = 3
        self.conv1d_out = (window_length - kernel_size + 1) * out_channels

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.conv1d_out, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1d.weight)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.relu(self.conv1d(state))
        x = x.contiguous().view(-1, self.conv1d_out)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, in_channels, window_length, action_size, seed, fc1, fc2):
        """Initialize parameters and build model.

        Args:
            in_channels (int): Dimension of each state
            window_length (int): Number of days in sliding window
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """

        out_channels = in_channels
        kernel_size = 3
        self.conv1d_out = (window_length - kernel_size + 1) * out_channels

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.conv1d_out + action_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1d.weight)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.relu(self.conv1d(state))
        x = x.contiguous().view(-1, self.conv1d_out)
        x = torch.cat((x, action), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
