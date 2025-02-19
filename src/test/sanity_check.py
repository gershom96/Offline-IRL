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
import yaml

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

# h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data_Feb18.h5"
model_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/runs/SCAND_test__2025-02-17 22:50:53/SCAND_test_epoch50.pth"
config_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/wandb/run-20250217_225053-jajyko8w/files/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

means = {
    "goal_distance": 24.5542033,
    "heading_error": -0.01489864,
    "velocity": 0.27514081,
    "omega": -0.0022823,
    "last_action": np.array([0.27517195, -0.00227932]),  # (2,)
    "preference_ranking": np.array([0.27514378440764425, -0.002269040021361245])  # (2,)
    }

stds = {
    "goal_distance": np.sqrt(259.24568856),
    "heading_error": np.sqrt(0.30830199),
    "velocity": np.sqrt(0.95987594),
    "omega": np.sqrt(0.02352064),
    "last_action": np.array([np.sqrt(0.95989415), np.sqrt(0.02352065)]),  # (2,)
    "preference_ranking": np.array([np.sqrt(0.9598264894150791), np.sqrt(0.023544157241549707)])  # (2,)
}

def revert_normalization(value, key, means, stds):
    val_mean = means[key]
    val_std = stds[key]
    return value * val_std + val_mean

dino_transform={
    "mean":[0.485, 0.456, 0.406],
    "std":[0.229, 0.224, 0.225]
}

# taken from SCANDPreferenceProcessor.create_action_space()
def create_action_space(last_action, expert_action):
    """
    creates actions space based on action from last timestep
    - expands and discretizes the action space based on delta and max values
    - makes sure the expert action (action from current timestep) is part of the action space
    :param last_action: action (v, w) from previous timestep
    :param expert_action: action (v, w) from current timestep
    - treated as 'expert action'
    :return:
    - v and w action spaces, and v, w expert action indices
    """
    # taken from SCANDPreferenceProcessor.__init__, values may have changed...
    v_max = 2
    w_max = 1.5
    d_v_max = 0.05
    d_w_max = 0.03
    n_v = 5
    n_w = 5

    last_v, last_w = last_action
    expert_v, expert_w = expert_action

    v_res = 2 * d_v_max / (n_v - 1)
    w_res = 2 * d_w_max / (n_w - 1)

    last_v = round(last_v / v_res) * v_res
    last_w = round(last_w / w_res) * w_res
    expert_v = round(expert_v / v_res) * v_res
    expert_w = round(expert_w / w_res) * w_res

    # Ensure velocities and omegas stay within bounds
    v_actions = np.clip(np.linspace(last_v - d_v_max, last_v + d_v_max, n_v), 0, v_max)
    w_actions = np.clip(np.linspace(last_w - d_w_max, last_w + d_w_max, n_w), -w_max, w_max)

    # Find expert indices
    expert_v_idx = np.argmin(np.abs(v_actions - expert_v))
    expert_w_idx = np.argmin(np.abs(w_actions - expert_w))
    # ensure expert actions are part of the action space
    v_actions[expert_v_idx] = expert_v
    w_actions[expert_w_idx] = expert_w

    if expert_v not in v_actions:
        raise Exception

    if expert_w not in w_actions:
        raise Exception

    return v_actions, w_actions, expert_v_idx, expert_w_idx

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

    print("image shape", images.shape)
    print("goal_distance shape", goal_distance.shape)
    # TODO: go permute this and see if the model still ranks the expert action the best
    order_array = np.arange(batch_size)
    np.random.shuffle(order_array)
    # velocity_s[0][0:5] = velocity[order_array[0]][0:5]
    velocity_s = velocity[order_array]
    # Forward pass
    t1 = time.time()
    reward = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
    t2 = time.time()

    print(f"Inference time: {t2 - t1:.4f} sec")
    # print("Reward Output Shape:", reward.shape)
    # print("Reward Output:", reward)
    # print("omega", omega[0][:5])
    # print("velocity", velocity[0][:5])
    reward = reward.cpu().detach().numpy()
    last_action = past_action.cpu().detach().numpy()
    best_actions = reward.argmax(axis=1)
    best_action_first = best_actions == 0
    print(f"{sum(best_action_first)} out of {len(best_actions)} has idx 0 as highest reward")
    count+=1

    if count==10:
        break

# TODO: go look at discretization schema of the create_preference_dataset_scand.py
# create a similar discretization schema

reward = reward.cpu().detach().numpy()
last_action = past_action.cpu().detach().numpy()
best_actions = reward.argmax(axis=1)

# get and un-normalize expert action
velocity_cvt = revert_normalization(velocity[:, 0], "velocity", means, stds)
omega_cvt = revert_normalization(omega[:, 0], "omega", means, stds)
v_omega_cvt = np.concatenate((velocity_cvt, omega_cvt), axis=-1)
last_action_cvt = (revert_normalization(last_action[:, 0, 0], "velocity", means, stds),
                   revert_normalization(last_action[:, 0, 1], "omega", means, stds))
last_action_cvt = np.stack(last_action_cvt, axis=1)

# which element in batch do we want
idx = 0
# discretize expert action to get original action_space:
v_actions, w_actions, expert_v_idx, expert_w_idx = create_action_space(last_action=last_action_cvt[idx], expert_action=v_omega_cvt[idx])

discr_ranked_actions = current_action
action_difference = current_action - v_omega_cvt
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