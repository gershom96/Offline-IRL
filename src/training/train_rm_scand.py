import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader, random_split
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand import RewardModelSCAND
from utils.plackett_luce_loss import PL_Loss

# user defined params;
project_name = "Offline-IRL"
exp_name = "SCAND_test"
h5_file = "/fs/nexus-scratch/gershom/IROS25/Datasets/scand_preference_data.h5"
checkpoint_dir = "/fs/nexus-scratch/gershom/IROS25/Offline-IRL/models/checkpoints"
BATCH_SIZE = 32 # 64 = 12GB VRAM, 32 = 6.9GB VRAM
LEARNING_RATE = 3e-4
NUM_QUERIES = 4
HIDDEN_DIM = 768
N_EPOCHS = 10
train_val_split = 0.8
num_workers = 4
batch_print_freq = 10
gradient_log_freq = 100
notes = "implementing wandb"
use_wandb = True
save_model = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
model = RewardModelSCAND(num_queries=NUM_QUERIES, hidden_dim=HIDDEN_DIM).to(device)
criterion = PL_Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
global_step = 0
start_time = time.time()
if use_wandb:
    if gradient_log_freq > 0:
        wandb.watch(model, log_freq=gradient_log_freq)

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0.0
    batch_count = 0

    # Training Loop (Per Batch Logging)
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
        predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size=len(images))  # (batch_size, 25)

        # Compute Loss
        loss = criterion(predicted_rewards)

        # Backpropagation
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        writer.add_scalar("charts/train_loss", loss.item(), global_step)
        batch_count += 1
        global_step += 1

        if batch_count % batch_print_freq == 0:  # Log every 10 batches
            SPS = global_step / (time.time() - start_time)
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] | Batch {batch_count} | Train Loss: {loss.item():.4f}, steps per second: {SPS:.3f}")
            writer.add_scalar("charts/SPS", SPS, global_step)
            writer.add_scalar("epoch", epoch, global_step)

    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar("charts/avg_train_loss", avg_train_loss, global_step)
    writer.add_scalar("epoch", epoch, global_step)

    # Validation Loop (At End of Each Epoch)
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