import numpy as np
import random
import copy
from collections import namedtuple, deque

from drl.model import Actor, Critic

import torch
import torch.nn.functional as nn_f
import torch.optim as optimum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, n_signals, window_length, n_assets, random_seed, lr_actor, lr_critic, buffer_size,
                 batch_size, tau, gamma, sigma, theta, fc1, fc2):
        """Initialize an Agent object.

        Args:
            n_signals (int): Number of signals per asset
            window_length (int): Number of days in sliding window
            n_assets (int): Number of assets in portfolio not counting cash
            random_seed (int): random seed
            lr_actor (float): initial learning rate for actor
            lr_critic (float): initial learning rate for critic
            batch_size (int): mini batch size
            buffer_size (int): replay buffer size
            gamma (float): discount factor
            tau (float): soft update of target parameters
            sigma (float): OU Noise standard deviation
            theta (float): OU Noise theta gain
            fc1 (int):  Size of 1st hidden layer
            fc2 (int):  Size of 2nd hidden layer
        """
        self.n_signals = n_signals
        self.n_assets = n_assets
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(n_signals, window_length, n_assets, random_seed, fc1, fc2).to(device)
        self.actor_target = Actor(n_signals, window_length, n_assets, random_seed, fc1, fc2).to(device)
        self.actor_optimizer = optimum.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(n_signals, window_length, n_assets, random_seed, fc1, fc2).to(device)
        self.critic_target = Critic(n_signals, window_length, n_assets, random_seed, fc1, fc2).to(device)
        self.critic_optimizer = optimum.Adam(self.critic_local.parameters(), lr=lr_critic)

        # Noise process
        self.noise = OUNoise(size=n_assets, seed=random_seed, sigma=sigma, theta=theta)

        # Replay memory
        self.memory = ReplayBuffer(n_assets, buffer_size, batch_size, random_seed)

    def step(self, state, actions, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, actions, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        self.actor_local.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_local.reset_parameters()
        self.critic_target.reset_parameters()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, done_values = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - done_values))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = nn_f.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, sigma, theta, mu=0.):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, n_assets, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.n_assets = n_assets
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(device)

        done_values = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        done_values = torch.from_numpy(done_values).float().to(device)

        return states, actions, rewards, next_states, done_values

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
