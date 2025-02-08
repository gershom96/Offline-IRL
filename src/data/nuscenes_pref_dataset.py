import h5py
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

class PreferenceDataset(Dataset):
    
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, "r")
        self.length = self.h5_file["preference_ranking"].shape[0]  # Total number of samples

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load data for the given index
        preference_ranking = self.h5_file["preference_ranking"][idx]   # Shape: (25, 2)
        goal_distance = self.h5_file["goal_distance"][idx]             # Shape: (1,)
        heading_error = self.h5_file["heading_error"][idx]             # Shape: (1,)
        velocity = self.h5_file["velocity"][idx]                       # Shape: (3,)
        rotation_rate = self.h5_file["rotation_rate"][idx]             # Shape: (3,)
        user_responses = self.h5_file["user_responses"][idx]           # Shape: (4,)

        # Convert to PyTorch tensors
        return {
            "preference_ranking": torch.tensor(preference_ranking, dtype=torch.float32),
            "goal_distance": torch.tensor(goal_distance, dtype=torch.float32),
            "heading_error": torch.tensor(heading_error, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
            "rotation_rate": torch.tensor(rotation_rate, dtype=torch.float32),
            "user_responses": torch.tensor(user_responses, dtype=torch.float32),
        }