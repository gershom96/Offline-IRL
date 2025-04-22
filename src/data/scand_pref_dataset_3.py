from typing import Any

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from io import BytesIO
import torchvision.transforms as transforms
import json

class SCANDPreferenceDataset(Dataset):
    def __init__(self, h5_file_path:str, scand_stats_path = '/home/gershom/Documents/GAMMA/IROS25/Repos/Offline-IRL/src/data_stats.json',transform = None):
        """
        Dataloader for annotated scan-d dataset
        annotated H5 file keys:
            ['goal_distance', 'heading', 'heading_error', 'image', 'last_action', 'omega', 'pos',
            'preference_ranking', 'sequence_info', 'user_responses', 'v']

        Args:
            h5_file_path (str): Path to the HDF5 dataset.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.h5_file_path = h5_file_path
        self.transform = transform  # Optional image transformations (e.g., resizing, normalization)

        # **DINOv2 Transformations (Resize + Normalize)**
        self.dino_transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            transforms.Resize((224, 224)),  # Input size to DINOv2
            transforms.ToTensor(),          
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])
        self.scand_stats_path = scand_stats_path

        with open(self.scand_stats_path, "r") as f:
            stats = json.load(f)
            means = stats["means"]
            stds = stats["stds"]

            self.means = {
                "goal_distance": float(means["goal_distance"]),
                "heading_error": float(means["heading_error"]),
                "velocity": float(means["velocity"]),
                "omega": float(means["omega"]),
                "last_action": np.array(means["last_action"]),  # (2,)
                "preference_ranking": np.array(means["preference_ranking"])  # (2,)
            }

            self.stds = {
                "goal_distance": float(stds["goal_distance"]),
                "heading_error": float(stds["heading_error"]),
                "velocity": float(stds["velocity"]),
                "omega": float(stds["omega"]),
                "last_action": np.array(stds["last_action"]),  # (2,)
                "preference_ranking": np.array(stds["preference_ranking"])  # (2,)
            }

        start_idx = 0
        self.group_indices = {}
        self.indices_to_group = []
        
        with h5py.File(self.h5_file_path, "r") as h5_file:  
            self.groups = list(h5_file.keys())
            
            self.sequence_info = h5_file["2"]["sequence_info"][:]

            for group in self.groups:
                if "image" not in h5_file[group]:  # Ensure "image" dataset exists
                    print(f"Skipping {group} (missing 'image' dataset)")
                    continue

                group_size = h5_file[group]["image"].shape[0]  # Get number of samples
                
                if group_size == 0:  # Check if the group is empty
                    print(f"Skipping {group} (empty dataset)")
                    continue  

                self.group_indices[group] = list(range(start_idx, start_idx + group_size))
                            
                # Assign indices to groups
                self.indices_to_group.extend([group] * group_size)
                start_idx += group_size

        self.length = len(self.indices_to_group)

        # Compute sampling weights only for **non-empty** groups
        self.weights_per_group = {
            group: self.length / (len(self.groups) * len(self.group_indices[group])) for group in self.group_indices
        }

        # Assign weights for each sample based on its group
        self.sample_weights = [self.weights_per_group[self.indices_to_group[i]] for i in range(self.length)]

    def __len__(self):
        return self.length

    def standardize(self, data, key):
        """Standardizes numerical values using precomputed mean and std."""
        return (data - self.means[key]) / (self.stds[key] + 1e-8)

    def load_image(self, image_data):
        """Loads image from HDF5 dataset."""
        if isinstance(image_data, np.ndarray):
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, (bytes, bytearray)):
            image_bytes = bytes(image_data)
        else:
            raise ValueError("Unsupported type for image_data: {}".format(type(image_data)))

        stream = BytesIO(image_bytes)
        image = Image.open(stream).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = self.dino_transform(image)

        return image

    def __getitem__(self, idx):

        # Data aggregation
        data: dict[str | Any, list[Any] | Tensor] = {
            # "goal_distance": [],
            # "heading_error": [],
            "velocity": [],
            "rotation_rate": [],
            "preference_ranking": [],
            "images": [],
            # "last_action": [],
            "pref_idx": []
        }

        group = self.indices_to_group[idx]  # Get the group name
        local_idx = self.group_indices[group].index(idx)  # Convert global index to local within the group
        
        with h5py.File(self.h5_file_path, "r", swmr=True) as h5_file:
            # Load shared state variables (Expand them for 25 actions **inside the dataset**)
            # goal_distance = h5_file[group]["goal_distance"][local_idx]  # (1,)
            # heading_error = h5_file[group]["heading_error"][local_idx]  # (1,)
            velocity = h5_file[group]["v"][local_idx]  # (1,)
            omega = h5_file[group]["omega"][local_idx]  # (1,)
            # last_action = h5_file[group]["last_action"][local_idx]  # (2,)
            # goal_distance = self.standardize(goal_distance, "goal_distance")
            # heading_error = self.standardize(heading_error, "heading_error")
            velocity = self.standardize(velocity, "velocity")
            omega = self.standardize(omega, "omega")
            # last_action = self.standardize(last_action, "last_action")

            # goal_distance = np.tile(goal_distance, (25, 1))  
            # heading_error = np.tile(heading_error, (25, 1))  
            velocity = np.tile(velocity, (25, 1))  
            omega = np.tile(omega, (25, 1))  
            # last_action = np.tile(last_action, (25, 1))  

            # Load per-action data
            preference_ranking = h5_file[group]["preference_ranking"][local_idx]  
            preference_ranking = self.standardize(preference_ranking, "preference_ranking")  

            # **Randomly shuffle the action order**
            perm_ =  np.random.permutation(25)
            perm = perm_.reshape(25,1)
            pref_idx = np.argsort(perm, axis = 0) 

            # **Apply permutation to all action-related tensors**
            preference_ranking = preference_ranking[perm_]  

            image = h5_file[group]["image"][local_idx]
            image = np.array(self.load_image(image))  # Load as NumPy
            data["images"].append(image) # Stack images

            # Store expanded data
            # data["goal_distance"].append(goal_distance)
            # data["heading_error"].append(heading_error)
            data["velocity"].append(velocity)
            data["rotation_rate"].append(omega)
            data["preference_ranking"].append(preference_ranking)
            # data["last_action"].append(last_action)
            data["pref_idx"].append(pref_idx)

            # Convert lists to tensors
            for key in data.keys():
                
                if key == "pref_idx":
                    data[key] = torch.from_numpy(np.array(data[key][0], dtype=np.int64))
                else:
                    data[key] = torch.from_numpy(np.array(data[key][0], dtype=np.float32))
               
            return data
