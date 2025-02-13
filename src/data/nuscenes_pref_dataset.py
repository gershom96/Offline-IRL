import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class NuScenesPreferenceDataset(Dataset):
    def __init__(self, h5_file_path, nuscenes_dataset_path, mode=4, time_window=1, transform = None):
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
        self.nuscenes_dataset_path = nuscenes_dataset_path
        self.length = self.h5_file["preference_ranking"].shape[0]
        self.mode = mode
        self.time_window = time_window
        self.sequence_info = self.h5_file["sequence_info"][:]  # (index_within_seq, seq_length)
        self.transform = transform  # Optional image transformations (e.g., resizing, normalization)

        # Camera indices
        self.camera_indices = {
            1: [0],            # Front Camera
            2: [0, 3],         # Front & Back
            3: [0, 3, 4, 5],   # Front, Back, Side-Back
            4: list(range(6))  # All 6 Cameras
        }

    def __len__(self):
        return self.length

    def load_image(self, image_path):
        """Loads an image from the given path."""
        image_path = image_path.decode("utf-8")  # Decode byte string to regular string
        # print(image_path)
        image = Image.open(os.path.join(self.nuscenes_dataset_path, image_path)).convert("RGB")  # Ensure 3 channels

        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        else:
            image = np.array(image, dtype=np.float32)        # Convert to NumPy array
            image = np.transpose(image, (2, 0, 1))     # Rearrange dimensions to [C, H, W] using NumPy

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
        # Time-series handling
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
            # Load dynamic data
            data["goal_distance"].append(self.h5_file["goal_distance"][i])
            data["heading_error"].append(self.h5_file["heading_error"][i])
            data["velocity"].append(self.h5_file["velocity"][i])
            data["rotation_rate"].append(self.h5_file["rotation_rate"][i])
            data["preference_ranking"].append(self.h5_file["preference_ranking"][i])
            data["last_action"].append(self.h5_file["last_action"][i])

            # Camera selection
            image_paths = self.h5_file["image_paths"][i]

            selected_cameras = self.camera_indices.get(self.mode, list(range(6)))
            selected_images = np.array([self.load_image(image_paths[j]) for j in selected_cameras])
            data["images"].append(selected_images)  # Stack images into a single tensor

        # Convert lists to tensors for dynamic data
        for key in data.keys():
            data[key] = torch.from_numpy(np.array(data[key], dtype=np.float32))  # Optimized conversion

        # print(data["images"].shape)

        return data
