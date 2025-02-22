from typing import Any

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from io import BytesIO
import torchvision.transforms as transforms
from tensordict.tensordict import TensorDict

class SCANDRLDataset(Dataset):
    def __init__(self, h5_file_path:str, device="cuda", transform = None):

        self.h5_file_path = h5_file_path
        if device == "gpu":
            device = "cuda"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.transform = transform  # Optional image transformations (e.g., resizing, normalization)

        start_idx = 0
        # h5 file keys
        # ['goal_distance', 'heading', 'heading_error', 'image', 'last_action', 'omega', 'pos', 'preference_ranking',
        #  'reward', 'sequence_info', 'user_responses', 'v']
        with h5py.File(self.h5_file_path, "r") as h5_file:
            self.len = len(h5_file['heading'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, "r", swmr=True) as h5_file:
            current_idx, end_idx = h5_file['sequence_info'][idx]
            terminal = False  # this is always false from the current setup!
            shifted = False
            # means this index is second to last frame in a scene...
            if current_idx == (end_idx - 1):
                shifted = True
                current_idx -= 1 # let's take the previous scene instead.

            img = h5_file['image'][current_idx]
            d_goal = h5_file['goal_distance'][current_idx]
            th_goal = h5_file['heading_error'][current_idx]
            v = h5_file['v'][current_idx]
            omega = h5_file['omega'][current_idx]
            current_actions = h5_file['preference_ranking'][current_idx]
            reward = h5_file['reward'][current_idx]

            next_idx = current_idx + 1
            img_next = h5_file['image'][next_idx]
            d_goal_next = h5_file['goal_distance'][next_idx]
            th_goal_next = h5_file['heading_error'][next_idx]
            v_next = h5_file['v'][next_idx]
            omega_next = h5_file['omega'][next_idx]

            tensordict = TensorDict({
                "observation": TensorDict({
                    "image": torch.Tensor(img).to(torch.float32),
                    "goal_distance": torch.Tensor(d_goal).to(torch.float32),
                    "heading_error": torch.Tensor(th_goal).to(torch.float32),
                }),
                "expert_action": TensorDict({
                    "v": torch.Tensor(v).to(torch.float32),
                    "omega": torch.Tensor(omega).to(torch.float32),
                }),
                "reward": torch.Tensor(reward).to(torch.float32),
                "terminal": torch.BoolTensor([terminal]),
                "next_state": TensorDict({
                    "image": torch.Tensor(img_next).to(torch.float32),
                    "goal_distance": torch.Tensor(d_goal_next).to(torch.float32),
                    "heading_error": torch.Tensor(th_goal_next).to(torch.float32),
                }),
                "current_actions": torch.Tensor(current_actions).to(torch.float32),
                "info": TensorDict({
                    "shifted": torch.BoolTensor([shifted])
                })
            },
            device=self.device,
            )
            return tensordict

