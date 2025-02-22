import sys
import os
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_model_scand import RewardModelSCAND
from data.scand_pref_dataset import SCANDPreferenceDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
import yaml
from scipy.stats import rankdata

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
# h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data_Feb18.h5"
model_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/runs/SCAND_test__2025-02-17 22:50:53/SCAND_test_epoch50.pth"
config_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/wandb/run-20250217_225053-jajyko8w/files/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

model = RewardModelSCAND(num_queries=config['num_queries']['value'],
                         num_heads=config['num_heads']['value'],
                         num_attn_stacks=config['addon_attn_stacks']['value'],
                         activation=config['activation_type']['value'],
                         # dropout=config['dropout_rate']['value'],
                         ).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-4)

checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
model.eval()

scand_dataset = SCANDPreferenceDataset(h5_file)

# Wrap in DataLoader
batch_size = 5
dataloader = DataLoader(scand_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
count = 0
# dataloader keys:
# ['goal_distance', 'heading_error', 'velocity', 'rotation_rate', 'preference_ranking', 'images', 'last_action']
# Iterate over dataset
for batch in dataloader:
    # Move batch data to device
    images = batch["images"].to(device)  # Shape: (batch_size, 3, 224, 224)
    goal_distance = batch["goal_distance"].to(device)  # Shape: (batch_size, 25, 1)
    heading_error = batch["heading_error"].to(device)  # Shape: (batch_size, 25, 1)
    velocity = batch["velocity"].to(device)  # Shape: (batch_size, 25, 1)
    omega = batch["rotation_rate"].to(device)  # Shape: (batch_size, 25, 1)
    past_action = batch["last_action"].to(device)  # Shape: (batch_size, 25, 2)
    current_action = batch["preference_ranking"].to(device)  # Shape: (batch_size, 25, 2)
    preference_scores = batch["preference_scores"].to(device)  # Shape: (batch_size, 25, 1)

    print("image shape", images.shape)
    print("goal_distance shape", goal_distance.shape)

    shuffle_array = np.arange(velocity.shape[1])
    np.random.shuffle(shuffle_array)
    unshuffle_array = np.zeros(velocity.shape[1]).astype(int)
    for i, shuffle_i in enumerate(shuffle_array):
        unshuffle_array[shuffle_i] = i

    # Forward pass
    t1 = time.time()
    reward_original = model(images, goal_distance, heading_error,
                     velocity, omega, past_action,
                     current_action, batch_size)

    reward_shuffle = model(images, goal_distance[:, shuffle_array], heading_error[:, shuffle_array],
                     velocity[:, shuffle_array], omega[:, shuffle_array], past_action[:, shuffle_array],
                     current_action[:, shuffle_array], batch_size)
    t2 = time.time()

    print(f"Inference time: {t2 - t1:.4f} sec")
    # print("Reward Output Shape:", reward.shape)
    # print("Reward Output:", reward)
    # print("omega", omega[0][:5])
    # print("velocity", velocity[0][:5])
    reward_unshuffle = reward_shuffle[:, unshuffle_array]
    reward_original = reward_original.cpu().detach().numpy()
    reward_unshuffle = reward_unshuffle.cpu().detach().numpy()
    preference_scores = preference_scores.cpu().detach().numpy().squeeze(-1)
    reward_ranks = rankdata(reward_unshuffle, axis=1)
    preference_ranks = rankdata(preference_scores, axis=1)
    avg_rank_diff_per_batch = np.abs(reward_ranks - preference_ranks).sum() / batch_size
    ideal_rank_diff_per_batch = np.abs(np.flip(np.arange(0, 25)) - preference_ranks).sum() / batch_size
    worst_case = np.abs(np.arange(0, 25) - preference_ranks).sum() / batch_size
    print(f"rank diff per batch, ideal: {ideal_rank_diff_per_batch:.3f}, avg: {avg_rank_diff_per_batch:.3f},  worst_case: {worst_case:.3f}")

    count+=1

    if count==10:
        break