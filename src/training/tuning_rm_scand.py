import torch
import torch.optim as optim
import sys
import os
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader, random_split
from data.scand_pref_dataset import SCANDPreferenceDataset
from utils.reward_model_scand import RewardModelSCAND
from utils.plackett_luce_loss import PL_Loss


def train(config=None):
    exp_name = "SCAND_tuning"
    h5_file = "/media/jim/Hard Disk/scand_data/rosbags/scand_preference_data.h5"
    notes = "CHANGE_ME"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Get the current time
    now = datetime.datetime.now()
    # Format the time as a string
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    run_name = f"{exp_name}__{timestamp}"

    wandb.init(
        notes=notes,
        sync_tensorboard=True,
        config=config,
        name=run_name,
        save_code=True,
    )
    # Load Dataset and Split
    dataset = SCANDPreferenceDataset(h5_file)
    train_size = int(wandb.config["train_val_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True,
                              num_workers=wandb.config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config["batch_size"], shuffle=False,
                            num_workers=wandb.config["num_workers"], pin_memory=True)


    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in wandb.config.items()])),
    )
    # Define Model, Loss, Optimizer
    model = RewardModelSCAND(num_queries=wandb.config["num_queries"], hidden_dim=wandb.config["hidden_dim"]).to(device)
    criterion = PL_Loss()
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    global_step = 0
    start_time = time.time()
    for epoch in range(wandb.config["epochs"]):
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
            predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action,
                                      current_action, batch_size=len(images))  # (batch_size, 25)

            # Compute Loss
            loss = criterion(predicted_rewards)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar("charts/train_loss", loss.item(), global_step)
            batch_count += 1
            global_step += 1

            if batch_count % wandb.config["batch_print_freq"] == 0:  # Log every 10 batches
                SPS = global_step / (time.time() - start_time)
                # print(
                    # f"Epoch [{epoch + 1}/{wandb.config["epochs"]}] | Batch {batch_count} | Train Loss: {loss.item():.4f}, steps per second: {SPS:.3f}")
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

                predicted_rewards = model(images, goal_distance, heading_error, velocity, omega, past_action,
                                          current_action, batch_size=len(images))
                loss = criterion(predicted_rewards)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("charts/avg_val_loss", avg_val_loss, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)

        # Print Epoch Results
        print(f"Epoch [{epoch + 1}/{wandb.config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)  # Adjust learning rate

        if wandb.config['save_model']:
            model_path = f"runs/{run_name}/{exp_name}.torch_model"
            torch.save(model.state_dict(), model_path)
            print(f"model saved to {model_path}")
    print("Training Complete!")
    writer.close()

# sweep configs
with open('../configs/default_sweep.yaml', 'r') as f:
    sweep_config = yaml.full_load(f)

sweep_id = wandb.sweep(sweep=sweep_config, project="scand_sweep")
wandb.agent(sweep_id, train, count=12)
