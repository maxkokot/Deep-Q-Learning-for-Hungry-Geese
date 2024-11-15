import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import \
    Observation, GreedyAgent, Configuration
from kaggle_environments import make

from collections import OrderedDict
from typing import Tuple, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer

from pytorch_lightning import LightningModule

from data.features import encode_state
from data.data import Experience, ReplayBuffer, RLDataset


class DQN(nn.Module):

    """Base Deep Q-Network.

    Args:
        n_rows: number of rows in the environment
        n_cols: number of cols in the environment
        n_ch: number of chanels in state representation

    """

    def __init__(self, n_rows: int = 7, n_cols: int = 11, n_ch: int = 17):

        super(DQN, self).__init__()
        fc1_size = (n_rows - 2 * 3) * (n_cols - 2 * 3) * 256
        self.conv1 = nn.Conv2d(n_ch, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(fc1_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class Goose:
    def __init__(self, replay_buffer, config) -> None:
        """Base Agent (Goose) class handling the interaction
        with the environment.

        Args:
            replay_buffer: replay buffer storing experiences
            config: enviroment configuration

        """
        self.config = config
        self.enemy_policy = GreedyAgent(
            Configuration({'rows': config['n_rows'],
                           'columns': config['n_cols']}))
        self.action_space = ["NORTH", "SOUTH", "WEST", "EAST"]

        self.init_env()
        self.replay_buffer = replay_buffer
        self.places = []
        self.surv_steps = []
        self.reset()

    def init_env(self):
        """Initializes enviroment objects."""
        geese = [None] + ["greedy"] * self.config['n_enemies']
        self.env = make("hungry_geese")
        self.trainer = self.env.train(geese)

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.trainer.reset()
        self.obs_list = []
        self.obs_list.append(self.state)
        self.enc_state = encode_state(self.obs_list, self.config)

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out
        using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action

        """
        if np.random.random() < epsilon:
            action_name = self.enemy_policy(Observation(self.state))
            action = self.action_space.index(action_name)

        else:
            state = self.enc_state.to(device)
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between
        the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action = self.get_action(net, epsilon, device)
        new_state, reward, \
            done, _ = self.trainer.step(self.action_space[action])
        reward = self._redefine_reward(reward, done)

        if done:
            score = self.env.state[0]['reward']
            place = sum([goose['reward'] > score
                         for goose in self.env.state]) + 1
            self.places.append(place)
            self.surv_steps.append(self.env.state[0]
                                   ['observation']['step'] + 1)

        else:
            self.obs_list.append(new_state)

        enc_new_state = encode_state(self.obs_list, self.config)
        exp = Experience(self.enc_state, action, reward, done, enc_new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        self.enc_state = enc_new_state

        if done:
            self.reset()

        return reward, done

    def _redefine_reward(self, reward: float, done: bool):
        """Change some reward values for more efficient learning

        Args:
            reward: initial reward
            done: if the episode is finished

        Returns:
            reward

        """
        # if done and reward == 0:  # collision
        #      reward = -1000
        if reward == 201:       # first step
            reward = 100
        elif reward == 101:     # eating
            reward = 150
        return reward


class DQNLightning(LightningModule):
    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-2,
        gamma: float = 0.99,
        plc_sync_rate: int = 5,
        tgt_sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        device: str = "cpu",
        config: dict = {}
    ) -> None:
        """Basic DQN Model.

        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            tgt_sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer
                             at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
            device: cpu or cuda
            config: enviroment configuration

        """
        super().__init__()
        self.save_hyperparameters()

        self.policy_net = DQN(n_rows=config['n_rows'],
                              n_cols=config['n_cols'],
                              n_ch=config['n_dims'],)
        self.target_net = DQN(n_rows=config['n_rows'],
                              n_cols=config['n_cols'],
                              n_ch=config['n_dims'],)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Goose(self.buffer, config)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_size)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment
        to initially fill up the replay buffer with experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        for _ in range(steps):
            self.agent.play_step(self.policy_net, epsilon=1.0)

    def accumulate(self, steps: int = 5, epsilon: int = 0) -> None:

        for _ in range(steps):

            # step through environment with agent
            reward, done = self.agent.play_step(self.policy_net, epsilon,
                                                self.hparams.device)
            self.episode_reward += reward

            if done:
                self.total_reward = self.episode_reward
                self.episode_reward = 0
                self.log("places", self.agent.places[-1])
                self.log("surv_steps", self.agent.surv_steps[-1],
                         prog_bar=True)
                self.log("total_reward", self.total_reward, prog_bar=True)

        return reward, done

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Passes in a state x through the network and gets
        the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.policy_net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.tensor,
                                        torch.tensor]) -> torch.tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss

        """
        states, actions, rewards, dones, next_states = batch
        state_action_values = self.policy_net(states)\
            .gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values *\
            self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor],
                      nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update
        the replay buffer. Then calculates loss based on
        the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        epsilon = self.get_epsilon(self.hparams.eps_start,
                                   self.hparams.eps_end,
                                   self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        reward, done = self.accumulate(self.hparams.plc_sync_rate, epsilon)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # Soft update of target network
        if self.global_step % self.hparams.tgt_sync_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.log("train_loss", loss)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.policy_net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset
        used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


def save_model(trainer, model_dir, model_name='model.ckpt'):
    path = os.path.join(model_dir, model_name)
    trainer.save_checkpoint(path)


def load_model(model_dir,
               model_name='model.ckpt'):
    path = os.path.join(model_dir, model_name)
    model = DQNLightning.load_from_checkpoint(checkpoint_path=path)
    return model
