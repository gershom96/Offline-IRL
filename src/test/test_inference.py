import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_model_scand import RewardModelSCAND
# from data.scand_pref_dataset_3 import SCANDPreferenceDataset3
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand_3 import RewardModelSCAND3
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
import yaml
from scipy.stats import rankdata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

# h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data_train.h5"
checkpoint_path = "/home/jim/Downloads/model_3_epoch_70.pth"
# model_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/checkpoints/Feb20_home/SCAND_test_epoch50.pth"
config_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/checkpoints/Feb20_home/config.yaml"

partial_load = False


def load_reward_model(config_path, model_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    reward_model = RewardModelSCAND3().to(device)
    model_optimizer = optim.AdamW(reward_model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    checkpoint = torch.load(model_path, map_location=device)
    reward_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    print(f"Loaded checkpoint from {model_path} at epoch {epoch}")
    reward_model.eval()
    return reward_model, model_optimizer


model, _ = load_reward_model(config_path, checkpoint_path)

scand_dataset = SCANDPreferenceDataset(h5_file)
# scand_dataset = SCANDPreferenceDataset3(h5_file)

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
    scores = batch["preference_scores"].to(device)  # Shape: (batch_size, 25, 2)
    perms = batch["pref_idx"].to(device)  # Shape: (batch_size, 25, 2)
    print(perms.shape)
    print(perms)

    # Forward pass
    t1 = time.time()
    reward = model(image, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
    t2 = time.time()
    ordered_reward = torch.gather(reward, 1, perms[:, :, 0])
    preferred_action_reward = ordered_reward[:, 0].cpu().detach().numpy()

    print(f"Inference time: {t2 - t1:.4f} sec")
    print("unscrambled first action rewards:", preferred_action_reward)

    count+=1

    if(count==10):
        break
    # break  # Only process one batch for testing