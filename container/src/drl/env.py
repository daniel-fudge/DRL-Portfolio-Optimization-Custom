""" Modified from https://github.com/awslabs/amazon-sagemaker-examples """

import os
import numpy as np
import pandas as pd

EPS = 1e-8


class PortfolioEnv:
    """ This class creates the financial market environment that the Agent interact with.

    The observations include a history of the signals with the given `window_length` ending at the current date.

    Args:
        steps (int):  Steps or days in an episode.
        trading_cost (float):  Cost of trade as a fraction.
        window_length (int):  How many past observations to return.
        start_date_index (int | None):  The date index in the signals and price arrays.
        prices_name (str):  CSV file name for the price history.

    Attributes:
        action_space (np.array):  [n_tickers]  The portfolio weighting not including cash.
        dates (np.array of np.datetime64):  [n_days] Dates for the signals and price history arrays.
        info_list (list):  List of info dictionaries for each step.
        n_signals (int):  Number of signals in each observation.
        n_tickers (int):  Number of tickers in the price history.
        observation_space (np.array)  [self.n_signals, window_length]  The signals with a window_length history.
        portfolio_value (float):  The portfolio value, starting with $1 in cash.
        gain (np.array):  [n_days, n_tickers] The relative price vector; tomorrow's / today's price.
        signals (np.array):  [n_signals, n_days, 1]  Signals that define the observable environment.
        start_date_index (int):  The date index in the signals and price arrays.
        step_number (int):  The step number of the episode.
        steps (int):  Steps or days in an episode.
        tickers (list of str):  The stock tickers.
        trading_cost (float):  Cost of trade as a fraction.
        window_length (int):  How many past observations to return.
    """

    def __init__(self, steps=505, trading_cost=0.0025, window_length=1, start_date_index=None,
                 prices_name='prices1.csv', output_path='/opt/ml/output/data/portfolio-management.csv'):
        """An environment for financial portfolio management."""

        # Initialize some local parameters
        self.csv = output_path
        self.info_list = list()
        self.portfolio_value = 1.0
        self.step_number = 0

        # Save some arguments as attributes
        self.trading_cost = trading_cost
        self.window_length = window_length

        # Read the stock data and convert to the relative price vector (gain)
        #   Note the raw prices have an extra day vs the signals to calculate gain
        src_folder = os.path.split(os.path.dirname(__file__))[0]
        raw_prices = pd.read_csv(os.path.join(src_folder, prices_name), index_col=0, parse_dates=True)
        self.tickers = raw_prices.columns.tolist()
        self.gain = np.hstack((np.ones((raw_prices.shape[0]-1, 1)), raw_prices.values[1:] / raw_prices.values[:-1]))
        self.dates = raw_prices.index.values[:-1]
        self.n_dates = self.dates.shape[0]
        self.n_tickers = len(self.tickers)
        self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)

        # Read the signals
        self.signals = pd.read_csv(os.path.join(src_folder, 'signals.csv'),
                                   index_col=0, parse_dates=True).T.values[:, :, np.newaxis]
        self.n_signals = self.signals.shape[0]

        # Define the action space as the portfolio weights where wn are [0, 1] for each asset not including cash
        self.action_space = np.empty(self.n_tickers)

        # Define the observation space, which are the signals
        self.observation_space = np.empty(self.n_signals * self.window_length)

        # Rest the environment
        self.start_date_index = start_date_index
        self.steps = steps
        self.reset()
        
    # -----------------------------------------------------------------------------------
    def step(self, action):
        """Step the environment.

        Args:
            action (np.array):  The desired portfolio weights [w0...].

        Returns:
            np.array:  [n_signals * window_length] The observation of the environment (state)
            float:  The reward received from the previous action.
            bool:  Indicates if the simulation is complete.
            dict:  Debugging information.
        """

        t = self.start_date_index + self.step_number  # t is when you make the trade
        w0 = self.weights                             # w0 is the portfolio weights before trading
        p0 = self.portfolio_value                     # p0 is the portfolio value before trading
        gain = self.gain[t]                              # y is the relative price vector; tomorrow's / today's price.

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
        state = self.signals[:, t0:t+1].flatten()

        # Save some information for debugging and plotting at the end
        r = gain.mean()
        if self.step_number == 0:
            market_value = r
        else:
            market_value = self.info_list[-1]["market_value"] * r 
        info = {"reward": reward, "portfolio_value": p1, 'date': self.dates[t],
                'steps': self.step_number, "market_value": market_value}
        self.info_list.append(info)

        # Check if finished and write to file
        done = False
        if (self.step_number >= self.steps) or (p1 <= 0):
            done = True
            pd.DataFrame(self.info_list).sort_values(by=['date']).to_csv(self.csv)

        self.step_number += 1

        return state, reward, done

    def reset(self):
        """Reset the environment to the initial state.

        Returns:
            np.array:  [n_signals * window_length] The first state observation
        """

        self.info_list = list()
        self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)
        self.portfolio_value = 1.0
        self.step_number = 0

        # Limit the number of steps
        self.steps = min(self.steps, self.n_dates - self.window_length - 1)

        # Control the start date
        if self.start_date_index is None:
            self.start_date_index = np.random.random_integers(self.window_length - 1, 
                                                              self.n_dates - self.steps - 1)
        else:     
            # noinspection PyTypeChecker
            self.start_date_index = np.clip(self.start_date_index,
                                            a_min=self.window_length - 1,
                                            a_max=self.n_dates - self.steps - 1)

        t = self.start_date_index + self.step_number
        t0 = t - self.window_length + 1
        state = self.signals[:, t0:t+1].flatten()

        return state
