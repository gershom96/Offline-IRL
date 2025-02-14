import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from io import BytesIO
import torchvision.transforms as transforms


class SCANDPreferenceDataset(Dataset):
    def __init__(self, h5_file_path, time_window=1, transform = None):
        """
        Args:
            h5_file_path (str): Path to the HDF5 dataset.
            mode (int): Determines which camera views to load.
                        1 - Front Camera
                        2 - Front & Back Cameras
                        3 - Front, Back & Side Cameras
                        4 - All 6 Cameras
            time_window (int): Number of sequential timesteps to include if time_series=True.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.h5_file = h5py.File(h5_file_path, "r")
        self.length = self.h5_file["preference_ranking"].shape[0]
        self.time_window = time_window
        self.sequence_info = self.h5_file["sequence_info"][:]  # (index_within_seq, seq_length)
        self.transform = transform  # Optional image transformations (e.g., resizing, normalization)

        # **DINOv2 Transformations (Resize + Normalize)**
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure images are resized properly
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])

        self.means = {
            "goal_distance": 24.5542033,  # Example: Replace with computed mean
            "heading_error": -0.01489864,
            "velocity": 0.27514081,
            "omega": -0.0022823,
            "last_action": np.array([0.27517195, -0.00227932]),  # (2,)
            "preference_ranking": np.array([0.27514378440764425, -0.002269040021361245])  # (2,)
        }

        self.stds = {
            "goal_distance": np.sqrt(259.24568856),  # Example: Replace with computed std
            "heading_error": np.sqrt(0.30830199),
            "velocity": np.sqrt(0.95987594),
            "omega": np.sqrt(0.02352064),
            "last_action": np.array([np.sqrt(0.95989415), np.sqrt(0.02352065)]),  # (2,)
            "preference_ranking": np.array([np.sqrt(0.9598264894150791), np.sqrt(0.023544157241549707)])  # (2,)
        }


    def __len__(self):
        return self.length

    def standardize(self, data, key):
        """Standardizes numerical values using precomputed mean and std."""
        return (data - self.means[key]) / (self.stds[key] + 1e-8)  # Avoid division by zero


    def load_image(self, image_data):
        # If image_data is a NumPy array (e.g. stored from h5 as np.uint8), convert it to bytes.
        if isinstance(image_data, np.ndarray):
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, (bytes, bytearray)):
            image_bytes = bytes(image_data)
        else:
            raise ValueError("Unsupported type for image_data: {}".format(type(image_data)))

        # Use BytesIO to treat the bytes as a file-like object for PIL.
        stream = BytesIO(image_bytes)
        image = Image.open(stream).convert("RGB")  # Decode the image and ensure 3 channels.
        
        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        else:
            image = self.dino_transform(image)
            
            # image = np.array(image, dtype=np.float32)        # Convert to NumPy array
            # image = np.transpose(image, (2, 0, 1))     # Rearrange dimensions to [C, H, W] using NumPy

        return image
    
    def get_time_series_indices(self, idx):
        """
        Handles time window sampling within sequence boundaries.
        """
        seq_idx, seq_len = self.sequence_info[idx]
        start_idx = max(0, idx - self.time_window + 1)
        start_idx = max(start_idx, idx - seq_idx)  # Prevent crossing sequence boundary

        indices = list(range(start_idx, idx + 1))
        while len(indices) < self.time_window:     # Pad if necessary
            indices.insert(0, start_idx)

        return indices

    def __getitem__(self, idx):
        indices = self.get_time_series_indices(idx)

        # Data aggregation
        data = {
            "goal_distance": [],
            "heading_error": [],
            "velocity": [],
            "rotation_rate": [],
            "preference_ranking": [],
            "images": [],
            "last_action": []
        }

        for i in indices:
            # Load shared state variables (Expand them for 25 actions **inside the dataset**)
            goal_distance = self.h5_file["goal_distance"][i]  # (1,)
            heading_error = self.h5_file["heading_error"][i]  # (1,)
            velocity = self.h5_file["v"][i]  # (1,)
            omega = self.h5_file["omega"][i]  # (1,)
            last_action = self.h5_file["last_action"][i]  # (2,)
            
            # **Standardize the numerical inputs**
            goal_distance = self.standardize(goal_distance, "goal_distance")
            heading_error = self.standardize(heading_error, "heading_error")
            velocity = self.standardize(velocity, "velocity")
            omega = self.standardize(omega, "omega")
            last_action = self.standardize(last_action, "last_action")

            # Expand shared state variables
            goal_distance = np.tile(goal_distance, (25, 1))  # (25, 1)
            heading_error = np.tile(heading_error, (25, 1))  # (25, 1)
            velocity = np.tile(velocity, (25, 1))  # (25, 1)
            omega = np.tile(omega, (25, 1))  # (25, 1)
            last_action = np.tile(last_action, (25, 1))  # (25, 2)

            # Load per-action data (Already 25 actions)
            preference_ranking = self.h5_file["preference_ranking"][i]  # (25, 2)
            preference_ranking = self.standardize(preference_ranking, "preference_ranking")  # Standardize

            # Load image
            image = self.h5_file["image"][i]
            image = np.array(self.load_image(image))  # Load as NumPy
            data["images"].append(image) # Stack images

            # Store expanded data
            data["goal_distance"].append(goal_distance)
            data["heading_error"].append(heading_error)
            data["velocity"].append(velocity)
            data["rotation_rate"].append(omega)
            data["preference_ranking"].append(preference_ranking)
            data["last_action"].append(last_action)

        # Convert lists to tensors
        for key in data.keys():
            if(self.time_window == 1):
                data[key] = torch.from_numpy(np.array(data[key][0], dtype=np.float32))
            else:
                data[key] = torch.from_numpy(np.array(data[key], dtype=np.float32))
        return data
