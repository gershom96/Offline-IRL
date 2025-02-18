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
import PIL
from trajectory_demo import draw_trajectory

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
model_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/runs/SCAND_test__2025-02-17 21:28:49/SCAND_test_epoch1.pth"
config_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/runs/SCAND_test__2025-02-17 21:28:49/files/config.yaml"
# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)
NUM_QUERIES = 8
NUM_HEADS = 8
ADDON_ATTN_STACKS = 2
activation_type = "relu"
dropout_rate = 0.0

# model = RewardModelSCAND(num_queries=config['num_queries']['value'],
#                          num_heads=config['num_heads']['value'],
#                          num_attn_stacks=config['addon_attn_stacks']['value'],
#                          activation=config['activation_type']['value'],
#                          dropout=config['dropout_rate']['value'],
#                          ).to(device)
model = RewardModelSCAND(num_queries=NUM_QUERIES,
                         num_heads=NUM_HEADS,
                         num_attn_stacks=ADDON_ATTN_STACKS,
                         activation=activation_type,
                         dropout=dropout_rate,
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
batch_size = 2
dataloader = DataLoader(scand_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
count = 0
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

    print("image shape", images.shape)
    print("goal_distance shape", goal_distance.shape)
    # Forward pass
    t1 = time.time()
    reward = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
    t2 = time.time()

    print(f"Inference time: {t2 - t1:.4f} sec")
    print("Reward Output Shape:", reward.shape)
    print("Reward Output:", reward)
    print("omega", omega[0][:5])
    print("velocity", velocity[0][:5])

    count+=1

    if count==1:
        break

# TODO: go look at discretization schema of the create_preference_dataset_scand.py
# create a similar discretization schema
# current action is the expert action??
reward = reward.cpu().detach().numpy()
best_actions = reward.argmax(axis=1)
v_omega = np.concatenate((velocity, omega), axis=-1)
action_difference = current_action - v_omega
action_difference = action_difference.cpu().detach().numpy()
action_diff_norm = np.linalg.norm(action_difference, axis=-1)
closest_actions = action_diff_norm.argmax(axis=1)
print("best action based on reward:")
print(best_actions)
print("actions closest to the current_action:")
print(closest_actions)
print(f"{sum(best_actions == closest_actions)} out of {len(best_actions)} correct!")





images = images.cpu()
omega = omega.cpu()
velocity = velocity.cpu()
first_img = images[0].numpy().transpose(1, 2, 0).astype(np.uint8)
# first_img = images[0].numpy().astype(np.uint8)
first_img = PIL.Image.fromarray(first_img)
first_img.show()
draw_trajectory(first_img, 0.0, 50, MODE="lines")
    # break  # Only process one batch for testing