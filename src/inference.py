import click
import logging

import pandas as pd
import numpy as np
import torch

from kaggle_environments import make

from data.features import encode_state
from models.model import load_model
from utils import load_config, add_n_dims

from matplotlib import pyplot as plt


def dql_agent(obs):

    """Wrapper for DQN model aimed to
       run and visualize game.
    """

    global obs_list, agent, policy_net, config
    obs_list.append(obs)
    agent.enc_state = encode_state(obs_list, config)

    with torch.no_grad():
        action = agent.get_action(policy_net,
                                  epsilon=0,
                                  device='cpu')
    action_name = agent.action_space[action]

    return action_name


def play_game(f_agent, render=True):

    """Runs and visualizes game.

    Args:
        f_agent: function representing your agent
        render: if visualization required

    Returns:
        place: place of your agent (if visualization is not required)
        win: True if your agent won (if visualization is not required)
    """

    global model, obs_list, agent, policy_net, config
    obs_list = []
    agent = model.agent
    policy_net = model.policy_net

    env = make("hungry_geese")
    geese = [f_agent] + ["greedy"] * config['n_enemies']

    run = env.run(geese)
    scores = [x['reward'] for x in run[-1]]
    place = len(scores) - list(np.argsort(scores)).index(0)
    win = place == 1

    if render:
        env.render(mode='ipython', width=500, height=500)
        print(f"your goose is white (placed {place})")
    else:
        return place, win


def play_n_games(f_agent, num_games=100):

    """Runs several games.

    Args:
        f_agent: function representing your agent
        num_games: number of games to play

    Returns:
        places: places of your agent
        wins: outcomes (wether your agent won the games)
    """

    wins = []
    places = []

    for i in range(num_games):
        place, win = play_game(f_agent, render=False)
        places.append(place)
        wins.append(win)

    return places, wins


def show_performance(f_agent, f_baseline="greedy", num_games=100):

    """Plots histograms of obtained places.

    Args:
        f_agent: function representing your agent
        f_baseline: baseline for comparison
        num_games: number of games to play
    """

    agent_places, agent_wins = play_n_games(f_agent, num_games)
    baseline_places, baseline_wins = play_n_games(f_baseline, num_games)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    pd.Series(baseline_places).hist(ax=axes[0])
    pd.Series(agent_places).hist(ax=axes[1])

    baseline_win_rate = np.sum(baseline_wins) / num_games
    dqn_win_rate = np.sum(agent_wins) / num_games

    axes[0].set_title(f"Greedy baseline. Win rate = {baseline_win_rate}")
    axes[1].set_title(f"DQN. Win rate = {dqn_win_rate}")

    plt.show()


def inference(model_config_path, env_config_path, single_game):

    global model, config
    model_config = load_config(model_config_path)
    env_config = load_config(env_config_path)
    config = add_n_dims(env_config)
    model_dir = model_config['model_dir']
    model = load_model(model_dir)

    if single_game:
        play_game(dql_agent)

    else:
        show_performance(dql_agent)


@click.command(name="inference")
@click.option('--model_config_path', default='../config/model_config.yaml')
@click.option('--env_config_path', default='../config/env_config.yaml')
@click.option('--single_game', default=False)
def inference_command(model_config_path, env_config_path, single_game):
    logger = logging.getLogger(__name__)
    logger.info('Launching a game')
    inference(model_config_path, env_config_path, single_game)
    logger.info('Game over')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    inference_command()
