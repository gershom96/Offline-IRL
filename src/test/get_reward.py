import sys
import os
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_model_scand import RewardModelSCAND
from data.scand_pref_dataset import SCANDPreferenceDataset
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
import yaml
from scipy.stats import rankdata

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
# h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data_Feb18.h5"
model_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/checkpoints/model_3_epoch_30.pth"
config_path = "/home/jim/Documents/Projects/Offline-IRL/src/configs/gershom_config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

model = RewardModelSCAND(num_queries=config['num_queries']['value'],
                         num_heads=config['num_heads']['value'],
                         num_attn_stacks=config['addon_attn_stacks']['value'],
                         activation=config['activation_type']['value'],
                         dropout=config['dropout']['value'],
                         ).to(device)
LR = float(config['learning_rate']['value'])
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# checkpoint = torch.load(model_path, weights_only=True)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# train_loss = checkpoint['train_loss']

checkpoint = torch.load(model_path, map_location=device)

print(f"\nTotal Layers in Checkpoint: {len(checkpoint['model_state_dict'])}")

total_layers = len(model.state_dict().keys())
missing_layers = [key for key in model.state_dict().keys() if key not in checkpoint['model_state_dict']]
print(f"\n Missing Layers (Expected in Model, but NOT in Checkpoint): {len(missing_layers)}")
missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

print("Missing Layers (not in checkpoint):", len(missing_layers), total_layers)
# print(checkpoint['optimizer_state_dict'].keys())
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
print(f"Loaded checkpoint from {model_path} at epoch {start_epoch}")

model.eval()

scand_dataset = SCANDPreferenceDataset(h5_file)
val_sampler = WeightedRandomSampler(weights=scand_dataset.sample_weights, num_samples=len(scand_dataset), replacement=True)

# Wrap in DataLoader
batch_size = 5
dataloader = DataLoader(scand_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, sampler=val_sampler)

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
    perms = batch["pref_idx"].to(device)
    batch_size = len(images)
    print("image shape", images.shape)
    print("goal_distance shape", goal_distance.shape)

    # Forward pass
    t1 = time.time()
    reward = model(images, goal_distance, heading_error,
                     velocity, omega, past_action,
                     current_action, batch_size)
    t2 = time.time()

    print(f"Inference time: {t2 - t1:.4f} sec")
    # print("Reward Output Shape:", reward.shape)
    # print("Reward Output:", reward)
    # print("omega", omega[0][:5])
    # print("velocity", velocity[0][:5])
    reward = reward.cpu().detach().numpy()
    reward_unshuffle = reward.cpu().detach().numpy()
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