import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import asdict, dataclass
from data.scand_rl_dataset import SCANDRLDataset
# from SCAND_IQL import TwinQ, ValueFunction, DeterministicPolicy, GaussianPolicy, ImplicitQLearning

# train_h5_path = "/media/jim/Hard Disk/scand_data/rosbags/scand_RL_data_train.h5"
train_h5_path = "/media/jim/Hard Disk/scand_data/rosbags/scand_RL_data_train_sample.h5"
# test_h5_path = "/media/jim/Hard Disk/scand_data/rosbags/scand_RL_data_test.h5"

@dataclass
class TrainConfig:
    # wandb project name
    project: str = "Offline-IRL"
    # wandb group name
    group: str = "IQL-D4RL"
    # wandb run name
    name: str = "IQL"
    # training dataset and evaluation environment
    env: str = "scand_rm"
    # discount factor
    discount: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.7
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 4 # 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    #  where to use dropout for policy network, optional
    actor_dropout: Optional[float] = None
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)



def collate_tensordicts(batch):
    return torch.stack(batch)

config = TrainConfig
training_data = SCANDRLDataset(train_h5_path, config.device)
# test_data = SCANDRLDataset(test_h5_path, device)

# q_network = TwinQ(state_dim, action_dim).to(config.device)
# v_network = ValueFunction(state_dim).to(config.device)


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=config.batch_size, collate_fn=collate_tensordicts)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_tensordicts)

for batch in train_dataloader:
    state_img = batch["observation"]["image"]
    state_goal_dist = batch["observation"]["goal_distance"]
    state_heading_error = batch["observation"]["heading_error"]
    print(batch)
    break