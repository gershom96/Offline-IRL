import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader, random_split
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand import RewardModelSCAND
from utils.plackett_luce_loss import PL_Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset and Split
h5_file = "/media/gershom/Media/Datasets/SCAND/scand_preference_data.h5"
dataset = SCANDPreferenceDataset(h5_file)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Define Model, Loss, Optimizer
model = RewardModelSCAND().to(device)
criterion = PL_Loss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 10

for epoch in range(num_epochs):
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
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1

        if batch_count % 10 == 0:  # Log every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}] | Batch {batch_count} | Train Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

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

    # Print Epoch Results
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)  # Adjust learning rate

print("Training Complete!")
