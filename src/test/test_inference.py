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

checkpoint_path = "/media/gershom/Media/Datasets/SCAND/model_3_epoch_30.pth"  
model = RewardModelSCAND3().to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model weights (Make sure the keys match)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()  # Set model to evaluation mode
print(f"Model checkpoint loaded from: {checkpoint_path}")

h5_file = "/media/gershom/Media/Datasets/SCAND/scand_preference_data_grouped_train.h5"
scand_dataset = SCANDPreferenceDataset3(h5_file)


train_sampler = WeightedRandomSampler(weights=scand_dataset.sample_weights, num_samples=len(scand_dataset), replacement=True)

# Wrap in DataLoader
batch_size = 1
dataloader = DataLoader(scand_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler = train_sampler)

count = 0
for batch in dataloader:
    # Move batch data to device
    image = batch["images"].to(device)  # Shape: (batch_size, 3, 224, 224)
    goal_distance = batch["goal_distance"].to(device)  # Shape: (batch_size, 25, 1)
    heading_error = batch["heading_error"].to(device)  # Shape: (batch_size, 25, 1)
    velocity = batch["velocity"].to(device)  # Shape: (batch_size, 25, 1)
    omega = batch["rotation_rate"].to(device)  # Shape: (batch_size, 25, 1)
    past_action = batch["last_action"].to(device)  # Shape: (batch_size, 25, 2)
    current_action = batch["preference_ranking"].to(device)  # Shape: (batch_size, 25, 2)
    perms = batch["pref_idx"].to(device) # Shape: (batch_size, 25, 1)

    t1 = time.time()
    reward = model(image, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
    t2 = time.time()

    print(f"Inference time: {t2 - t1:.4f} sec")
    # print("Reward Output Shape:", reward.shape)

    # print(perms.shape)
    reordered_rewards = torch.gather(reward, dim=1, index=perms.squeeze(-1))
    reordered_actions = torch.gather(current_action, dim=1, index=perms.expand(-1, -1, 2))

    print("Rewards Reshaped:", reordered_rewards)
    # print("Actions Reshaped:", reordered_actions)
    # # Print action space
    # print("Sample Action Space:")
    # print(current_action[0].cpu().numpy())  # Print actions for batch 0

    # count += 1
    # if count == 10:
    #     break  # Stop after 10 samples
    # Unnormalize actions
    v_mean = 0.27514081
    v_std = np.sqrt(0.95987594)
    w_mean = -0.0022823
    w_std = np.sqrt(0.02352064)

    unnormalized_actions = reordered_actions.clone().detach().cpu().numpy()
    unnormalized_actions[..., 0] = unnormalized_actions[..., 0] * v_std + v_mean  # Velocity
    unnormalized_actions[..., 1] = unnormalized_actions[..., 1] * w_std + w_mean  # Omega

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

    # Display unnormalized actions
    print("Unnormalized Actions:", unnormalized_actions)

    plt.show()