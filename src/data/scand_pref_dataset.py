import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from io import BytesIO

class SCANDPreferenceDataset(Dataset):
    def __init__(self, h5_file_path, scand_dataset_path, time_window=1, transform = None):
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
        self.scand_dataset_path = scand_dataset_path
        self.length = self.h5_file["preference_ranking"].shape[0]
        self.time_window = time_window
        self.sequence_info = self.h5_file["sequence_info"][:]  # (index_within_seq, seq_length)
        self.transform = transform  # Optional image transformations (e.g., resizing, normalization)

    def __len__(self):
        return self.length

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
            "image": []
        }

        for i in indices:
            # Load dynamic data
            data["goal_distance"].append(self.h5_file["goal_distance"][i])
            data["heading_error"].append(self.h5_file["heading_error"][i])
            data["velocity"].append(self.h5_file["velocity"][i])
            data["rotation_rate"].append(self.h5_file["rotation_rate"][i])
            data["preference_ranking"].append(self.h5_file["preference_ranking"][i])

            # Camera selection
            image = self.h5_file["images"][i]

            image = np.array(self.load_image(image))
            data["images"].append(image)  # Stack images into a single tensor

        # Convert lists to tensors for dynamic data
        for key in data.keys():
            data[key] = torch.from_numpy(np.array(data[key], dtype=np.float32))  # Optimized conversion

        # print(data["images"].shape)

        return data
