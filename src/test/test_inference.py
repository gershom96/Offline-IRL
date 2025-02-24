import sys
import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_model_scand_3 import RewardModelSCAND3  # Ensure correct model class
from data.scand_pref_dataset_3 import SCANDPreferenceDataset3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

checkpoint_path = "/media/gershom/Media/Datasets/SCAND/model_3_epoch_70.pth"  
model = RewardModelSCAND3().to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model weights (Make sure the keys match)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()  # Set model to evaluation mode
print(f"Model checkpoint loaded from: {checkpoint_path}")

h5_file = "/media/gershom/Media/Datasets/SCAND/scand_preference_data_grouped_test.h5"
scand_dataset = SCANDPreferenceDataset3(h5_file)


train_sampler = WeightedRandomSampler(weights=scand_dataset.sample_weights, num_samples=len(scand_dataset), replacement=True)

# Wrap in DataLoader
batch_size = 1
dataloader = DataLoader(scand_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler = train_sampler)


def create_action_space(v_center, w_center, n_v=5, n_w=5, v_step=0.025, w_step=0.015, v_max=2.0, w_max=1.5):

    v_actions = np.clip(np.linspace(v_center - ((n_v - 1) / 2) * v_step, 
                                    v_center + ((n_v - 1) / 2) * v_step, n_v), 0, v_max)

    w_actions = np.clip(np.linspace(w_center - ((n_w - 1) / 2) * w_step, 
                                    w_center + ((n_w - 1) / 2) * w_step, n_w), -w_max, w_max)

    return v_actions, w_actions

count = 0

v_mean = scand_dataset.means["velocity"]
v_std = scand_dataset.stds["velocity"]
w_mean = scand_dataset.means["omega"]
w_std = scand_dataset.stds["omega"]

for batch in dataloader:
    # Move batch data to device
    image = batch["images"].to(device)  # Shape: (batch_size, 3, 224, 224)
    goal_distance = batch["goal_distance"].to(device)  # Shape: (batch_size, 25, 1)
    heading_error = batch["heading_error"].to(device)  # Shape: (batch_size, 25, 1)
    velocity = batch["velocity"].to(device)  # Shape: (batch_size, 25, 1)
    omega = batch["rotation_rate"].to(device)  # Shape: (batch_size, 25, 1)
    past_action = batch["last_action"].to(device)  # Shape: (batch_size, 25, 2)
    perms = batch["pref_idx"].to(device) # Shape: (batch_size, 25, 1)

    inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
                         std=[1/0.229, 1/0.224, 1/0.225])
    ])
    # Convert image tensor to PIL Image
    image_tensor = image.squeeze(0).cpu()  # Remove batch dimension
    image_tensor = inv_normalize(image_tensor)
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    image_np = (image_np * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]

    # Display image and actions
    plt.figure(figsize=(8, 6))
    plt.imshow(image_np)
    plt.title("Image with Action Space")
    plt.axis("off")
    plt.draw()
    plt.pause(0.1)

    past_action[...,0] = past_action[...,0]* v_std + v_mean
    past_action[...,1] = past_action[...,1]* w_std + w_mean
    while True:
        
        print(f"Actual past action: {past_action[0][0]}")
        past_v = float(input("Enter past_v: ").strip())
        past_w = float(input("Enter past_w: ").strip())

        past_action = np.zeros((1,25,2))
        past_action[:,:,0] = past_v
        past_action[:,:,1] = past_w

        past_action = torch.tensor(past_action, dtype=torch.float32).to(device)
        # User-defined action center
        v_center = float(input("Center v: ").strip())
        w_center = float(input("Center w: ").strip())

        # Generate the action space
        L_v, L_w = create_action_space(v_center, w_center)
        # print(L_v, L_w)

        
        # Combine v and w into a (25, 2) action matrix
        current_action = np.array([[(v, w) for v in L_v for w in L_w]])  # Shape: (25, 2)
        current_action = torch.tensor(current_action, dtype=torch.float32).to(device)

        current_action[..., 0] = (current_action[..., 0] - v_mean) / v_std  # Normalize velocity
        current_action[..., 1] = (current_action[..., 1] - w_mean) / w_std  # Normalize omega

        t1 = time.time()
        reward = model(image, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
        t2 = time.time()

        reward = reward.cpu()
        current_action.cpu()
        sorted_indices = torch.argsort(reward.squeeze(), descending=True)

        # print("Reward Shape:", reward.shape)  # Should be (25,)
        # print("Sorted Indices:", sorted_indices.shape)  # Should be (25,)
        print(f"Inference time: {t2 - t1:.4f} sec")
        
        # count += 1
        # if count == 10:
        #     break  # Stop after 10 samples

        unnormalized_actions = current_action.clone().detach().cpu().numpy()
        unnormalized_actions[..., 0] = unnormalized_actions[..., 0] * v_std + v_mean  # Velocity
        unnormalized_actions[..., 1] = unnormalized_actions[..., 1] * w_std + w_mean  # Omega

        
        sorted_rewards = reward[:,sorted_indices]  # Apply sorting to rewards
        sorted_actions = unnormalized_actions[:,sorted_indices,:]  # Apply sorting to actions
        # Display unnormalized actions
        print("Sorted Rewards:", sorted_rewards)

        print("Sorted Actions:", sorted_actions)

        test_again = bool(int(input("Test Again? : ").strip()))

        if(not test_again):
            break
    plt.close()