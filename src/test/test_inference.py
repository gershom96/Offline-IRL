import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_model_scand import RewardModelSCAND
from data.scand_pref_dataset import SCANDPreferenceDataset
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

h5_file = "/media/gershom/Media/Datasets/SCAND/scand_preference_data.h5"

model = RewardModelSCAND().to(device)
scand_dataset = SCANDPreferenceDataset(h5_file)

# Wrap in DataLoader
batch_size = 4
dataloader = DataLoader(scand_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
count = 0
# Iterate over dataset
for batch in dataloader:
    # Move batch data to device
    image = batch["images"].to(device)  # Shape: (batch_size, 3, 224, 224)
    goal_distance = batch["goal_distance"].to(device)  # Shape: (batch_size, 25, 1)
    heading_error = batch["heading_error"].to(device)  # Shape: (batch_size, 25, 1)
    velocity = batch["velocity"].to(device)  # Shape: (batch_size, 25, 1)
    omega = batch["rotation_rate"].to(device)  # Shape: (batch_size, 25, 1)
    past_action = batch["last_action"].to(device)  # Shape: (batch_size, 25, 2)
    current_action = batch["preference_ranking"].to(device)  # Shape: (batch_size, 25, 2)

    # print(image.shape)
    # print(goal_distance.shape)
    # Forward pass
    t1 = time.time()
    reward = model(image, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
    t2 = time.time()

    print(f"Inference time: {t2 - t1:.4f} sec")
    print("Reward Output Shape:", reward.shape)
    print("Reward Output:", reward)

    count+=1

    if(count==10):
        break
    # break  # Only process one batch for testing