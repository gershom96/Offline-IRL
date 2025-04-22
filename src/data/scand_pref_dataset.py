from typing import Any

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from io import BytesIO
import torchvision.transforms as transforms


class SCANDPreferenceDataset(Dataset):
    def __init__(self, h5_file_path:str, mode:int = 1, time_window:int = 1, transform = None):
        """
        Dataloader for annotated scan-d dataset
        annotated H5 file keys:
            ['goal_distance', 'heading', 'heading_error', 'image', 'last_action', 'omega', 'pos',
            'preference_ranking', 'sequence_info', 'user_responses', 'v']

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
        self.h5_file_path = h5_file_path
        self.transform = transform  # Optional image transformations (e.g., resizing, normalization)

        # **DINOv2 Transformations (Resize + Normalize)**
        self.dino_transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            transforms.Resize((224, 224)),  # Ensure images are resized properly
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])

        self.means = {
            "goal_distance": 24.8459378,
            "heading_error": -0.01240172,
            "velocity": 0.91916239,
            "omega": -0.00177657,
            "last_action": np.array([0.91924777, -0.00177915]),  # (2,)
            "preference_ranking": np.array([0.9191205964026605, -0.0017618069756301425])  # (2,)
        }

        self.stds = {
            "goal_distance": np.sqrt(269.07137664),
            "heading_error": np.sqrt(0.31416974),
            "velocity": np.sqrt(0.63170535),
            "omega": np.sqrt(0.02270943),
            "last_action": np.array([np.sqrt(0.63164339), np.sqrt(0.02270509)]),  # (2,)
            "preference_ranking": np.array([np.sqrt(0.631699838954012), np.sqrt(0.022741062629058388)])  # (2,)
        }

        start_idx = 0
        self.group_indices = {}
        self.indices_to_group = []  # Store the group each index belongs to

        with h5py.File(self.h5_file_path, "r") as h5_file:

            self.sequence_info = h5_file["2"]["sequence_info"][:]  # (index_within_seq, seq_length)

            self.groups = list(h5_file.keys())

            for group in self.groups:
                group_size = h5_file[group]["goal_distance"].shape[0]
                self.group_indices[group] = list(range(start_idx, start_idx + group_size))

                # Assign indices to groups
                self.indices_to_group.extend([group] * group_size)
                start_idx += group_size

        self.length = len(self.indices_to_group) # Total dataset size

        # Compute sampling weights
        self.weights_per_group = {
            group: self.length / (len(self.groups) * len(self.group_indices[group])) for group in self.groups
        }

        # Assign weights for each sample based on its group
        self.sample_weights = [self.weights_per_group[self.indices_to_group[i]] for i in range(self.length)]

    def __len__(self):
        return self.length

    def standardize(self, data, key):
        """Standardizes numerical values using precomputed mean and std."""
        return (data - self.means[key]) / (self.stds[key] + 1e-8)  # Avoid division by zero


    def load_image(self, image_data):
        """Loads image from HDF5 dataset."""
        if isinstance(image_data, np.ndarray):
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, (bytes, bytearray)):
            image_bytes = bytes(image_data)
        else:
            raise ValueError("Unsupported type for image_data: {}".format(type(image_data)))

        stream = BytesIO(image_bytes)
        image = Image.open(stream).convert("RGB")  # Decode the image and ensure 3 channels.

        if self.transform:
            image = self.transform(image)
        else:
            image = self.dino_transform(image)
            
            # image = np.array(image, dtype=np.float32)        # Convert to NumPy array
            # image = np.transpose(image, (2, 0, 1))     # Rearrange dimensions to [C, H, W] using NumPy

        return image
    
    def get_time_series_indices(self, idx):
        """Handles time window sampling within sequence boundaries."""
        seq_idx, seq_len = self.sequence_info[idx]
        start_idx = max(0, idx - self.time_window + 1)
        start_idx = max(start_idx, idx - seq_idx)  # Prevent crossing sequence boundary

        indices = list(range(start_idx, idx + 1))
        while len(indices) < self.time_window:
            indices.insert(0, start_idx)

        return indices

    def __getitem__(self, idx):

        # Data aggregation
        data: dict[str | Any, list[Any] | Tensor] = {
            "goal_distance": [],
            "heading_error": [],
            "velocity": [],
            "rotation_rate": [],
            "preference_ranking": [],
            "preference_scores": [],
            "images": [],
            "last_action": [],
            "pref_idx": []
        }

        group = self.indices_to_group[idx]  # Get the group name
        local_idx = self.group_indices[group].index(idx)  # Convert global index to local within the group

        with h5py.File(self.h5_file_path, "r", swmr=True) as h5_file:
            # Load shared state variables (Expand them for 25 actions **inside the dataset**)
            goal_distance = h5_file[group]["goal_distance"][local_idx]  # (1,)
            heading_error = h5_file[group]["heading_error"][local_idx]  # (1,)
            velocity = h5_file[group]["v"][local_idx]  # (1,)
            omega = h5_file[group]["omega"][local_idx]  # (1,)
            last_action = h5_file[group]["last_action"][local_idx]  # (2,)

            # Standardize the numerical inputs
            goal_distance = self.standardize(goal_distance, "goal_distance")
            heading_error = self.standardize(heading_error, "heading_error")
            velocity = self.standardize(velocity, "velocity")
            omega = self.standardize(omega, "omega")
            last_action = self.standardize(last_action, "last_action")

            # Expand shared state variables to (25, x)
            goal_distance = np.tile(goal_distance, (25, 1))
            heading_error = np.tile(heading_error, (25, 1))
            velocity = np.tile(velocity, (25, 1))
            omega = np.tile(omega, (25, 1))
            last_action = np.tile(last_action, (25, 1))

            # Load per-action data
            preference_ranking = h5_file[group]["preference_ranking"][local_idx]
            preference_ranking = self.standardize(preference_ranking, "preference_ranking")
            preference_scores = h5_file[group]["preference_scores"][local_idx]  # (25, 1)

            # **Randomly shuffle the action order**
            perm_ = np.random.permutation(25)
            perm = perm_.reshape(25, 1)
            pref_idx = np.argsort(perm, axis=0)

            # **Apply permutation to all action-related tensors**
            preference_ranking = preference_ranking[perm_]

            # Load image
            image = h5_file[group]["image"][local_idx]
            image = np.array(self.load_image(image))  # Load as NumPy
            data["images"].append(image) # Stack images

            # Store expanded data, ALL NORMALIZED!!
            data["goal_distance"].append(goal_distance)
            data["heading_error"].append(heading_error)
            data["velocity"].append(velocity)
            data["rotation_rate"].append(omega)
            data["preference_ranking"].append(preference_ranking)
            data["preference_scores"].append(preference_scores)
            data["last_action"].append(last_action)
            data["pref_idx"].append(pref_idx)

            # Convert lists to tensors
            for key in data.keys():

                if key == "pref_idx":
                    data[key] = torch.from_numpy(np.array(data[key][0], dtype=np.int64))
                else:
                    data[key] = torch.from_numpy(np.array(data[key][0], dtype=np.float32))


            return data
