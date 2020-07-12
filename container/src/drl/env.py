""" Modified from https://github.com/awslabs/amazon-sagemaker-examples """

import os
import numpy as np
import pandas as pd

EPS = 1e-8


class PortfolioEnv:
    """ This class creates the financial market environment that the Agent interact with.

    The observations include a history of the signals with the given `window_length` ending at the current date.

    Args:
        trading_cost (float):  Cost of trade as a fraction.
        window_length (int):  How many past observations to return.
        prices_name (str):  CSV file name for the price history.

    Attributes:
        action_space (np.array):  [n_tickers]  The portfolio weighting not including cash.
        dates (np.array of np.datetime64):  [n_days] Dates for the signals and price history arrays.
        n_signals (int):  Number of signals in each observation.
        n_tickers (int):  Number of tickers in the price history.
        observation_space (np.array)  [batch_size, n_signals, window_length]  The signals with a window_length history.
        market_value (float):  The market value, starting at $1.
        portfolio_value (float):  The portfolio value, starting with $1 in cash.
        gain (np.array):  [n_days, n_tickers] The relative price vector; tomorrow's / today's price.
        signals (np.array):  [n_signals, n_days, 1]  Signals that define the observable environment.
        start_day (int):  The start date index in the signals and price arrays.
        step_number (int):  The step number of the episode.
        tickers (list of str):  The stock tickers.
        trading_cost (float):  Cost of trade as a fraction.
        window_length (int):  How many past observations to return.
        weights (np.array):  [1 + n_tickers]  The portfolio asset weighting starting with cash.
    """

    def __init__(self, trading_cost, window_length, prices_name):
        """An environment for financial portfolio management."""

        # Initialize some local parameters
        self.market_value = 1.0
        self.portfolio_value = 1.0
        self.step_number = 0
        self.start_day = 0

        # Save some arguments as attributes
        self.trading_cost = trading_cost
        self.window_length = window_length

        # Read the stock data and convert to the relative price vector (gain)
        #   Note the raw prices have an extra day vs the signals to calculate gain
        src_folder = os.path.split(os.path.dirname(__file__))[0]
        raw_prices = pd.read_csv(os.path.join(src_folder, prices_name), index_col=0, parse_dates=True)
        self.tickers = raw_prices.columns.tolist()
        self.gain = np.hstack((np.ones((raw_prices.shape[0]-1, 1)), raw_prices.values[1:] / raw_prices.values[:-1]))
        self.dates = raw_prices.index.values
        self.n_dates = self.dates.shape[0] - 1
        self.n_tickers = len(self.tickers)
        self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)

        # Read the signals
        self.signals = pd.read_csv(os.path.join(src_folder, 'signals.csv'),
                                   index_col=0, parse_dates=True).T.values[np.newaxis, :, :]
        self.n_signals = self.signals.shape[1]

        # Define the action space as the portfolio weights where wn are [0, 1] for each asset not including cash
        self.action_space = np.empty(self.n_tickers)

        # Define the observation space, which are the signals [batch_size, n_signals, windows_length]
        self.observation_space = np.empty([1, self.n_signals, self.window_length])
        
    # -----------------------------------------------------------------------------------
    def step(self, action):
        """Step the environment.

        Args:
            action (np.array):  The desired portfolio weights [w0...].

        Returns:
            np.array:  [n_signals * window_length] The observation of the environment (state).
            float:  The reward received from the previous action.
        """

        t = self.start_day + self.step_number  # t is when you make the trade
        w0 = self.weights                      # w0 is the portfolio weights before trading
        p0 = self.portfolio_value              # p0 is the portfolio value before trading
        gain = self.gain[t]                    # gain is the relative price vector; tomorrow's / today's price.

        # Force the new weights (w1) to (0.0, 1.0) and sum weights = 1, note 1st weight is cash
        #   w0_post is the desired portfolio weighting after the trades but still at time t
        w0_post = np.clip(action, a_min=0, a_max=1)
        w0_post = np.insert(w0_post, 0, np.clip(1 - w0_post.sum(), a_min=0, a_max=1))
        w0_post = w0_post / w0_post.sum()

        # Calculate the loss due to trading costs
        loss = p0 * self.trading_cost * (np.abs(w0_post - w0)).sum()

        # Calculate the weights after the price evolution, w1 is the weights at t + 1
        w1 = (gain * w0_post) / (np.dot(gain, w0_post) + EPS)

        # Calculate the portfolio value at t + 1
        p1 = p0 * np.dot(gain, w0_post) - loss
        p1 = np.clip(p1, 0, np.inf)                         # Limit portfolio to zero (busted)
        reward = np.log((p1 + EPS) / (p0 + EPS))            # log rate of return

        # Save weights and portfolio value for next iteration
        self.weights = w1
        self.portfolio_value = p1

        # Observe the new environment (state)
        t0 = t - self.window_length + 1
        state = self.signals[:, :, t0:t+1]

        # Save market value and increment the step number
        self.market_value *= gain.mean()
        self.step_number += 1

        return state, reward

    def reset(self, epoch_start, market_value=1.0, portfolio_value=1.0, weights=None):
        """Reset the environment to the initial state.

        Args:
            epoch_start (int):  The epoch start date index.
            market_value (float):  The market value.
            portfolio_value (float):  The portfolio value.
            weights (np.array):  [1 + n_tickers]  The portfolio asset weighting starting with cash.

        Returns:
            np.array:  [n_signals * window_length] The first state observation
        """

        self.start_day = epoch_start
        if weights is None:
            self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)
        else:
            self.weights = weights
        self.market_value = market_value
        self.portfolio_value = portfolio_value
        self.step_number = 0

        t = self.start_day + self.step_number
        t0 = t - self.window_length + 1
        state = self.signals[:, :, t0:t+1]

        return state
