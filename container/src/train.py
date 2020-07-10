"""
This script trains and saves the model and plots its performance.
"""

import ast
import argparse
from collections import deque
import logging
import numpy as np
import os
import platform
from drl.ddpg_agent import Agent
from drl.env import PortfolioEnv
from time import time
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    logger.info("GPU activated.")
else:
    device = torch.device("cpu")
    logger.info("CPU activated.")


# ***************************************************************************************
def make_plot(output_dir, show=False):
    """Makes a pretty training plot call score.png.

    Args:
        output_dir (str):  Location to save output.
        show (bool):  If True, show the image.  If False, save the image.
    """

    import matplotlib.pyplot as plt

    target = 0.5

    # Load the previous scores and calculated running mean of 100 runs
    # ---------------------------------------------------------------------------------------
    with np.load(os.path.join(output_dir, 'scores.npz')) as data:
        scores = data['arr_0']
    cum_sum = np.cumsum(np.insert(scores, 0, 0))
    rolling_mean = (cum_sum[100:] - cum_sum[:-100]) / 100

    # Make a pretty plot
    # ---------------------------------------------------------------------------------------
    plt.figure()
    x_max = len(scores)
    y_min = scores.min() - 1
    x = np.arange(x_max)
    plt.scatter(x, scores, s=2, c='k', label='Raw Scores', zorder=4)
    plt.plot(x[99:], rolling_mean, lw=2, label='Rolling Mean', zorder=3)
    plt.scatter(x_max, rolling_mean[-1], c='g', s=40, marker='*', label='Episode {}'.format(x_max), zorder=5)
    plt.plot([0, x_max], [target, target], lw=1, c='grey', ls='--', label='Target Score = {}'.format(target), zorder=1)
    plt.plot([x_max, x_max], [y_min, rolling_mean[-1]], lw=1, c='grey', ls='--', label=None, zorder=2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.xlim([0, x_max + 5])
    plt.ylim(bottom=0)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'scores.png'), dpi=200)
    plt.close()


# ***************************************************************************************
def train(epochs, max_t, output_dir, model_dir):
    """This function trains the given agent in the given environment.

    Args:
        epochs (int): Maximum number of training epochs
        max_t (int): Maximum number of time steps per episode
        output_dir (str):  Location to save output.
        model_dir (str):  Location to save checkpoints and final model.
    """

    scores = list()
    scores_window = deque(maxlen=100)
    brain_name = env.brain_names[0]
    start_time = time()
    i_episode = epochs
    for i_episode in range(1, epochs + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = 0
        for t in range(max_t):
            actions = [agents[i].act(state=states[i]) for i in range(2)]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done_values = env_info.local_done
            for i in range(2):
                agents[i].step(states, actions, rewards, next_states, done_values)
            states = next_states
            score += max(rewards)
            if np.any(done_values):
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        if i_episode % 100 == 0:
            print('Episode {} Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5:
            tmp_str = '\nEnvironment solved in {:d} episodes!  Average Score: {:.2f}'
            print(tmp_str.format(i_episode, np.mean(scores_window)))
            break
    print('{:d} training episodes completed.'.format(i_episode))
    mean_score = np.mean(scores_window)
    print('{:.2f} average score.'.format(mean_score))
    duration = (time() - start_time)/60
    print('{:.2f} minutes of training.'.format(duration))
    print('{:.2f} training objective.'.format(i_episode - 1000*mean_score + 10*duration))

    # Save models weights and scores
    # -----------------------------------------------------------------------------------
    for p in [p for p in [model_dir, output_dir] if not os.path.isdir(p)]:
        os.mkdir(p)
    for i in range(2):
        torch.save(agents[i].actor_target.state_dict(),
                   os.path.join(model_dir, 'checkpoint_actor_{}.pth'.format(i + 1)))
        torch.save(agents[i].critic_target.state_dict(),
                   os.path.join(model_dir, 'checkpoint_critic_{}.pth'.format(i + 1)))
    np.savez(os.path.join(output_dir, 'scores.npz'), scores)


# ***************************************************************************************
if __name__ == '__main__':

    # Read the arguments
    # -----------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # These are general setting
    parser.add_argument('--prices_name', type=str, default='prices1.csv', metavar='P',
                        help='the csv file name containing the price history (default: prices1.csv)')

    # These are hyperparameters that could be tuned
    parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                        help='number of total epochs to run (default: 2000)')
    parser.add_argument('--max_t', type=int, default=1000, metavar='T',
                        help='max number of time steps per epoch (default: 1000)')
    parser.add_argument('--fc1', type=int, default=128, metavar='FC1',
                        help='size of 1st hidden layer (default: 128)')
    parser.add_argument('--fc2', type=int, default=64, metavar='FC2',
                        help='size of 2bd hidden layer (default: 64)')
    parser.add_argument('--lr_actor', type=float, default=0.001, metavar='LRA',
                        help='initial learning rate for actor (default: 0.001)')
    parser.add_argument('--lr_critic', type=float, default=0.001, metavar='LRC',
                        help='initial learning rate for critic (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='BS',
                        help='mini batch size (default: 256)')
    parser.add_argument('--buffer_size', type=int, default=int(1e5), metavar='BFS',
                        help='replay buffer size (default: 10,000)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                        help='discount factor (default: 0.9)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='TAU',
                        help='soft update of target parameters (default: 0.001)')
    parser.add_argument('--sigma', type=float, default=0.01, metavar='S',
                        help='OU Noise standard deviation (default: 0.01)')

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'],
                        help='where the trained model should be saved')
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'],
                        help='where SageMaker will place the training data')
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'],
                        help='where miscellaneous files should be saved')
    args = parser.parse_args()

    # Setup the training environment
    # -----------------------------------------------------------------------------------
    logger.info('Setting up the environment.')
    env = PortfolioEnv(prices_name=args.prices_name)

    # size of each action
    action_size = env.action_space
    logger.info('Size of action space: {}'.format(action_size))

    # examine the state space
    state_size = env.observation_space
    logger.info('State space per agent: {}'.format(state_size))

    # Create the reinforcement learning agents
    # -----------------------------------------------------------------------------------
    agents = [Agent(state_size=state_size, action_size=action_size, random_seed=42, lr_actor=args.lr_actor,
                    lr_critic=args.lr_critic, batch_size=args.batch_size, buffer_size=args.buffer_size,
                    gamma=args.gamma, tau=args.tau, sigma=args.sigma, fc1=args.fc1, fc2=args.fc2) for _ in range(2)]

    # Perform the training
    # -----------------------------------------------------------------------------------
    logger.info('Training the agent.')
    start = time()
    train(epochs=args.epochs, max_t=args.max_t, output_dir=args.output_dir, model_dir=args.model_dir)

    logger.info("Training Time:  {:.1f} minutes".format((time() - start)/60.0))

    # Make some pretty plots
    # -----------------------------------------------------------------------------------
    logger.info('Make training plot called scores.png.')
    make_plot(output_dir=args.output_dir)
