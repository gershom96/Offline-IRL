import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader, random_split
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand import RewardModelSCAND
from utils.reward_model_scand_2 import RewardModelSCAND2

from utils.plackett_luce_loss import PL_Loss

# user defined params;
project_name = "Offline-IRL"
exp_name = "SCAND_test"
h5_file = "/fs/nexus-scratch/gershom/IROS25/Datasets/scand_preference_data.h5"
checkpoint_dir = "/fs/nexus-scratch/gershom/IROS25/Offline-IRL/src/models/checkpoints"
BATCH_SIZE = 256 
LEARNING_RATE = 5e-5
NUM_QUERIES = 4
HIDDEN_DIM = 768
N_EPOCHS = 200
train_val_split = 0.8
num_workers = 4
batch_print_freq = 5
gradient_log_freq = 50
notes = "implementing wandb"
use_wandb = True
save_model = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset and Split
dataset = SCANDPreferenceDataset(h5_file)
train_size = int(train_val_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

run_config = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": N_EPOCHS,
    "num_queries": NUM_QUERIES,
    "hidden_dim": HIDDEN_DIM,
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
model = RewardModelSCAND(num_queries=NUM_QUERIES).to(device)
criterion = PL_Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Load from latest checkpoint (if available)
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Find latest checkpoint
        latest_checkpoint_path = "/fs/nexus-scratch/gershom/IROS25/Offline-IRL/src/models/checkpoints/model2_epoch_40.pth"
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {latest_checkpoint_path} at epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No previous checkpoint found. Starting fresh.")
else:
    start_epoch = 0
    print("Checkpoint directory does not exist. Starting fresh.")

global_step = 0
start_time = time.time()

if use_wandb and gradient_log_freq > 0:
    wandb.watch(model, log_freq=gradient_log_freq)

# Training Loop
for epoch in range(start_epoch, N_EPOCHS):  # Start from checkpointed epoch
    model.train()
    train_loss = 0.0
    batch_count = 0

    for batch in train_loader:
        images = batch["images"].to(device)
        goal_distance = batch["goal_distance"].to(device)
        heading_error = batch["heading_error"].to(device)
        velocity = batch["velocity"].to(device)
        omega = batch["rotation_rate"].to(device)
        past_action = batch["last_action"].to(device)
        current_action = batch["preference_ranking"].to(device)

        optimizer.zero_grad()

        # Forward Pass
        predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size=len(images))

        # Compute Loss
        loss = criterion(predicted_rewards)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        writer.add_scalar("charts/train_loss", loss.item(), global_step)
        batch_count += 1
        global_step += 1

        if batch_count % batch_print_freq == 0:
            SPS = global_step / (time.time() - start_time)
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] | Batch {batch_count} | Train Loss: {loss.item():.4f}, steps per second: {SPS:.3f} | LR: {optimizer.param_groups[0]['lr']}")
            writer.add_scalar("charts/SPS", SPS, global_step)
            writer.add_scalar("epoch", epoch, global_step)

    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar("charts/avg_train_loss", avg_train_loss, global_step)
    writer.add_scalar("epoch", epoch, global_step)

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            goal_distance = batch["goal_distance"].to(device)
            heading_error = batch["heading_error"].to(device)
            velocity = batch["velocity"].to(device)
            omega = batch["rotation_rate"].to(device)
            past_action = batch["last_action"].to(device)
            current_action = batch["preference_ranking"].to(device)

            predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size=len(images))
            loss = criterion(predicted_rewards)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("charts/avg_val_loss", avg_val_loss, global_step)
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)

    # Print Epoch Results
    print(f"Epoch [{epoch+1}/{N_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)  # Adjust learning rate

    if (epoch + 1) % 20 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")

        # Save only trainable parameters (excluding frozen ones)
        trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

print("Training Complete!")
writer.close()
