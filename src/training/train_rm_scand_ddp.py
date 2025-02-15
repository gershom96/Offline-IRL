import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader, random_split
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand import RewardModelSCAND
from utils.plackett_luce_loss import PL_Loss


def setup(rank, world_size):
    """Initialize the distributed training process"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy the process group after training"""
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    # Move model to GPU
    device = torch.device(f"cuda:{rank}")
    model = RewardModelSCAND().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    criterion = PL_Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Load Dataset and Split
    h5_file = "/fs/nexus-scratch/gershom/IROS25/Datasets/scand_preference_data.h5"
    dataset = SCANDPreferenceDataset(h5_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Distributed Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_sampler.set_epoch(epoch)  # Ensure shuffling per epoch

        for batch in train_loader:
            images = batch["images"].to(device)
            goal_distance = batch["goal_distance"].to(device)
            heading_error = batch["heading_error"].to(device)
            velocity = batch["velocity"].to(device)
            omega = batch["rotation_rate"].to(device)
            past_action = batch["last_action"].to(device)
            current_action = batch["preference_ranking"].to(device)

            optimizer.zero_grad()
            predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size=len(images))
            loss = criterion(predicted_rewards)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

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

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of GPUs
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
