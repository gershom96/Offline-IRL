import torch
import torch.optim as optim
import sys
import os
import time
import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand import RewardModelSCAND
from utils.plackett_luce_loss import PL_Loss

from torchinfo import summary
from scipy.stats import rankdata
import numpy as np

# user defined params;
project_name = "Offline-IRL"
exp_name = "SCAND_test"
# h5_file = "/media/jim/7C846B9E846B5A22/scand_data/rosbags/scand_preference_data.h5"
h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
checkpoint_dir = "/home/jim/Documents/Projects/Offline-IRL/src/training/checkpoints"
latest_checkpoint_path = "/home/jim/Documents/Projects/Offline-IRL/src/training/runs/SCAND_test__2025-02-17 22:50:53/SCAND_test_epoch50.pth"
# h5_file = "/fs/nexus-scratch/gershom/IROS25/Datasets/scand_preference_data.h5"
# checkpoint_dir = "/fs/nexus-scratch/gershom/IROS25/Offline-IRL/models/checkpoints"
load_files = False
BATCH_SIZE = 96 # 128 = 23.1GB 64 = 12GB, 32 = 6.9GB VRAM
LEARNING_RATE = 3e-4
NUM_QUERIES = 12
NUM_HEADS = 8
N_EPOCHS = 100
ADDON_ATTN_STACKS = 2
ACTIVATION_TYPE = "relu"
DROPOUT_RATE = 0.1
train_val_split = 0.8
num_workers = 1
batch_print_freq = 10
gradient_log_freq = 100
save_model_freq = 5
# notes = "jim-desktop new loss fn"
notes = "gammawks03"
use_wandb = False
save_model = False
save_model_summary = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Load Dataset and Split
train_dataset = SCANDPreferenceDataset(h5_file)
val_dataset = SCANDPreferenceDataset(h5_file)

train_sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=len(train_dataset), replacement=True)
val_sampler = WeightedRandomSampler(weights=val_dataset.sample_weights, num_samples=len(val_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, sampler = train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, sampler = val_sampler)

run_config = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": N_EPOCHS,
    "num_queries": NUM_QUERIES,
    "num_heads": NUM_HEADS,
    "addon_attn_stacks" : ADDON_ATTN_STACKS,
    "activation_type" : ACTIVATION_TYPE,
    "dropout_rate" : DROPOUT_RATE,
    "train_val_split": train_val_split,
    "num_workers": num_workers,
    "save_model": save_model,
    "batch_print_freq": batch_print_freq,
}
# Get the current time
now = datetime.datetime.now()

# Format the time as a string
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
run_name = f"{exp_name}__{timestamp}"
if use_wandb:
    import wandb
    wandb.init(
        project=project_name,
        notes=notes,
        sync_tensorboard=True,
        config=run_config,
        name=run_name,
        save_code=True,
    )
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in run_config.items()])),
)

# Define Model, Loss, Optimizer
model = RewardModelSCAND(num_queries=NUM_QUERIES, num_heads=NUM_HEADS,
                         num_attn_stacks=ADDON_ATTN_STACKS, activation=ACTIVATION_TYPE,
                         dropout=DROPOUT_RATE).to(device)
if save_model_summary:
    model_summary = summary(model)
    f = open(f"runs/{run_name}/model_summary.txt", "w")
    f.write(str(model_summary))
    f.close()
# save config!
with open(f'runs/{run_name}/config.yaml', 'w') as file:
    yaml.dump(run_config, file)

criterion = PL_Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
start_epoch = 0

global_step = 0

start_time = time.time()

if use_wandb and gradient_log_freq > 0:
    wandb.watch(model, log_freq=gradient_log_freq)

# Training Loop
for epoch in range(start_epoch, N_EPOCHS):  # Start from checkpointed epoch
    model.train()
    train_loss = 0.0
    batch_count = 0
    rank_diff = 0
    # Training Loop (Per Batch Logging)
    for batch in train_loader:
        images = batch["images"].to(device)
        goal_distance = batch["goal_distance"].to(device)
        heading_error = batch["heading_error"].to(device)
        velocity = batch["velocity"].to(device)
        omega = batch["rotation_rate"].to(device)
        past_action = batch["last_action"].to(device)
        current_action = batch["preference_ranking"].to(device)
        preference_scores = batch["preference_scores"]
        perms = batch["pref_idx"].to(device)
        batch_size = len(images)

        optimizer.zero_grad()

        # Forward Pass
        predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size=len(images))  # (batch_size, 25)

        # Compute Loss
        loss = criterion(predicted_rewards, perms)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        writer.add_scalar("charts/train_loss", loss.item(), global_step)

        reward = predicted_rewards.cpu().detach().numpy()
        preference_scores = preference_scores.cpu().detach().numpy().squeeze(-1)
        reward_ranks = rankdata(reward, axis=1)
        preference_ranks = rankdata(preference_scores, axis=1)
        avg_rank_diff_per_batch = np.abs(reward_ranks - preference_ranks).sum() / len(reward)
        rank_diff += avg_rank_diff_per_batch
        writer.add_scalar("charts/avg_rank_diff_per_batch", avg_rank_diff_per_batch, global_step)

        batch_count += 1
        global_step += 1

        if batch_count % batch_print_freq == 0:
            SPS = global_step / (time.time() - start_time)
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] | Batch {batch_count} | Train Loss: {loss.item():.4f}, steps per second: {SPS:.3f} | LR: {optimizer.param_groups[0]['lr']}")
            writer.add_scalar("charts/SPS", SPS, global_step)
            writer.add_scalar("epoch", epoch, global_step)

    avg_train_loss = train_loss / len(train_loader)
    avg_rank_diff = train_loss / len(train_loader)
    writer.add_scalar("charts/avg_train_loss", avg_train_loss, global_step)
    writer.add_scalar("charts/avg_train_rank_diff", avg_rank_diff, global_step)
    writer.add_scalar("epoch/avg_train_loss", avg_train_loss, epoch)
    writer.add_scalar("epoch", epoch, global_step)

    # Validation Loop
    model.eval()
    val_loss = 0.0
    rank_diff = 0.0
    rank_diff_shuffled = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            goal_distance = batch["goal_distance"].to(device)
            heading_error = batch["heading_error"].to(device)
            velocity = batch["velocity"].to(device)
            omega = batch["rotation_rate"].to(device)
            past_action = batch["last_action"].to(device)
            current_action = batch["preference_ranking"].to(device)
            preference_scores = batch["preference_scores"]

            shuffle_array = np.arange(velocity.shape[1])
            np.random.shuffle(shuffle_array)
            unshuffle_array = np.zeros(velocity.shape[1]).astype(int)
            for i, shuffle_i in enumerate(shuffle_array):
                unshuffle_array[shuffle_i] = i

            predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size=len(images))
            shuffled_rewards = model(images, goal_distance[:, shuffle_array], heading_error[:, shuffle_array],
                                   velocity[:, shuffle_array], omega[:, shuffle_array], past_action[:, shuffle_array],
                                   current_action[:, shuffle_array], len(images))

            loss = criterion(predicted_rewards, perms)

            reward_original = predicted_rewards.cpu().detach().numpy()
            reward_unshuffle = shuffled_rewards[:, unshuffle_array].cpu().detach().numpy()
            preference_scores = preference_scores.cpu().detach().numpy().squeeze(-1)

            reward_ranks = rankdata(reward_original, axis=1)
            reward_ranks_unshuffle = rankdata(reward_unshuffle, axis=1)
            preference_ranks = rankdata(preference_scores, axis=1)

            avg_rank_diff_per_batch = np.abs(reward_ranks - preference_ranks).sum() / len(reward_ranks)
            unshuffled_avg_rank_diff = np.abs(reward_ranks_unshuffle - preference_ranks).sum() / len(reward_ranks_unshuffle)

            rank_diff += avg_rank_diff_per_batch
            rank_diff_shuffled += unshuffled_avg_rank_diff


            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_rank_diff = rank_diff / len(val_loader)
    avg_rank_diff_shuffled = rank_diff_shuffled / len(val_loader)
    writer.add_scalar("charts/avg_val_loss", avg_val_loss, global_step)
    writer.add_scalar("charts/avg_val_rank_diff", avg_rank_diff, global_step)
    writer.add_scalar("charts/avg_val_rank_diff_shuffled", avg_rank_diff_shuffled, global_step)
    writer.add_scalar("epoch/avg_val_loss", avg_val_loss, epoch)
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)

    # Print Epoch Results
    print(f"Epoch [{epoch+1}/{N_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(epoch)  # Adjust learning rate


    if (epoch + 1) % save_model_freq == 0:
        # checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        checkpoint_path = f"runs/{run_name}/{exp_name}_epoch{epoch + 1}.pth"

        # Save only trainable parameters (excluding frozen ones)
        # trainable_state_dict = {k: v for k, v in model.state_dict().items() if "vision_model" not in k}

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

print("Training Complete!")
writer.close()
