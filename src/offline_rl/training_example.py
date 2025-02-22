import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from data.scand_rl_dataset import SCANDRLDataset

# train_h5_path = "/media/jim/Hard Disk/scand_data/rosbags/scand_RL_data_train.h5"
train_h5_path = "/media/jim/Hard Disk/scand_data/rosbags/scand_RL_data_train_sample.h5"
# test_h5_path = "/media/jim/Hard Disk/scand_data/rosbags/scand_RL_data_test.h5"

batch_size = 4
device = "cuda"
training_data = SCANDRLDataset(train_h5_path, device)
# test_data = SCANDRLDataset(test_h5_path, device)
def collate_tensordicts(batch):
    return torch.stack(batch)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_tensordicts)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_tensordicts)

for batch in train_dataloader:
    state_img = batch["observation"]["image"]
    state_goal_dist = batch["observation"]["goal_distance"]
    state_heading_error = batch["observation"]["heading_error"]
    print(batch)
    break