import h5py
import os
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image

from utils.reward_model_scand_3 import RewardModelSCAND3  # Ensure correct model class

def quaternion_to_yaw(q):
    """Converts quaternion to yaw angle."""
    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    return np.arctan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """Normalizes angle to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class SCANDRLProcessor:
    def __init__(self, output_h5_path_expert, output_h5_path_other, reward_model_path, v_max=2, w_max=1.5, d_v_max=0.05, d_w_max=0.03, n_v=5, n_w=5, scand_stats_path = '../scand_data_stats.json'):
        
        self.scand_stats_path = scand_stats_path

        with open(self.scand_stats_path, "r") as f:
            stats = json.load(f)
            means = stats["means"]
            stds = stats["stds"]
        
            self.v_mean = float(means["velocity"])
            self.v_std =  float(stds["velocity"])
            self.w_mean = float(means["omega"])
            self.w_std = float(stds["omega"])

        self.v_max = v_max
        self.w_max = w_max
        self.d_v_max = d_v_max
        self.d_w_max = d_w_max
        self.n_v = n_v
        self.n_w = n_w
        self.last_action = None  # Initialize this later for each scene
        self.tau_1 = 0.32
        self.tau_2 = 1

        self.output_h5_path_expert = output_h5_path_expert
        self.output_h5_path_other = output_h5_path_other
        self.verbose = False
        self.outlier_window = 40
        self.transform = False

        self.v_res = 2 * self.d_v_max / (self.n_v - 1)
        self.w_res = 2 * self.d_w_max / (self.n_w - 1)
        # Create consolidated HDF5 file if it doesn't exist
        if not os.path.exists(output_h5_path_expert):
            with h5py.File(output_h5_path_expert, "w") as f:
                pass
                
        if not os.path.exists(output_h5_path_other):
            with h5py.File(output_h5_path_other, "w") as f:
                pass
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model = RewardModelSCAND3()

        checkpoint = torch.load(reward_model_path, map_location=self.device)
        self.reward_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.reward_model.to(self.device)
        self.reward_model.eval()

        self.means = {
            "goal_distance": 25.00290765,
            "heading_error": -0.01144954,
            "velocity": 1.2836152,
            "omega": -0.00149038,
            "last_action": np.array([1.28372591, -0.00149611]),  # (2,)
            "preference_ranking": np.array([1.2835427186288364, -0.00147476722583554])  # (2,)
        }

        self.stds = {
            "goal_distance": np.sqrt(275.01400778),
            "heading_error": np.sqrt(0.31931658),
            "velocity": np.sqrt(0.07844775),
            "omega": np.sqrt(0.02225859),
            "last_action": np.array([np.sqrt(0.07829221), np.sqrt(0.02225181)]),  # (2,)
            "preference_ranking": np.array([np.sqrt(0.0785323071548214), np.sqrt(0.022286368831911443)])  # (2,)
        }

        # **DINOv2 Transformations (Resize + Normalize)**
        self.dino_transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            transforms.Resize((224, 224)),  # Ensure images are resized properly
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])


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

        return image
    
    def extract_rewards(self, goal_distance, heading_error, velocity, omega, last_action, current_action, image):
        
        goal_distance = self.standardize(goal_distance, "goal_distance")
        heading_error = self.standardize(heading_error, "heading_error")
        velocity = self.standardize(velocity, "velocity")
        omega = self.standardize(omega, "omega")
        last_action = self.standardize(last_action, "last_action")

        goal_distance = np.tile(goal_distance, (25, 1))  
        heading_error = np.tile(heading_error, (25, 1))  
        velocity = np.tile(velocity, (25, 1))  
        omega = np.tile(omega, (25, 1))  
        last_action = np.tile(last_action, (25, 1))

        current_action = self.standardize(current_action, "preference_ranking")  
        image = np.array(self.load_image(image))
        
        goal_distance = torch.from_numpy(np.array(goal_distance, dtype=np.float32)).to(self.device)
        heading_error = torch.from_numpy(np.array(heading_error, dtype=np.float32)).to(self.device)
        velocity = torch.from_numpy(np.array(velocity, dtype=np.float32)).to(self.device)
        omega = torch.from_numpy(np.array(omega, dtype=np.float32)).to(self.device)
        last_action = torch.from_numpy(np.array(last_action, dtype=np.float32)).to(self.device)
        current_action = torch.from_numpy(np.array(current_action, dtype=np.float32)).to(self.device)
        image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0).to(self.device)

        # print(goal_distance.shape)
        
        with torch.no_grad():

            rewards = self.reward_model(image, goal_distance, heading_error, velocity, omega, last_action, current_action, 1)

        # print(current_action)
        # print(rewards)
        # raise Exception
        return rewards   
    
    def create_action_space(self, last_action, expert_action, discretize = True):

        # print(last_action)
        last_v, last_w = last_action
        expert_v, expert_w = expert_action

        
        if(discretize):
            # print(last_v)
            last_v = round(last_v / self.v_res) * self.v_res
            last_w = round(last_w / self.w_res) * self.w_res
            expert_v = round(expert_v / self.v_res) * self.v_res
            expert_w = round(expert_w / self.w_res) * self.w_res

        # Ensure velocities and omegas stay within bounds
        v_actions = np.clip(np.linspace(last_v - self.d_v_max, last_v + self.d_v_max, self.n_v), 0, self.v_max)
        w_actions = np.clip(np.linspace(last_w - self.d_w_max, last_w + self.d_w_max, self.n_w), -self.w_max, self.w_max)

        # Find expert indices
        expert_v_idx = np.argmin(np.abs(v_actions - expert_v))
        expert_w_idx = np.argmin(np.abs(w_actions - expert_w))

        v_actions[expert_v_idx] = expert_v
        w_actions[expert_w_idx] = expert_w
        
        if(expert_v not in v_actions):
            raise Exception

        if(expert_w not in w_actions):
            raise Exception
        
        return v_actions, w_actions, expert_v_idx, expert_w_idx
    
    def get_outliers(self, h5_filepath, scene):

        range_dict = {
            "v+w": [],
            "v": [],
            "w": []
        }
        with h5py.File(h5_filepath, "r") as h5file:

            velocities = h5file["v"][:]
            omegas = h5file["omega"][:]
            n_samples = h5file["image"].shape[0]

            v_w_idx = -1
            v_idx = -1
            w_idx = -1

            count_outliers = 0
            for i in range(n_samples):
                velocity_outlier = (velocities[i]>self.v_mean + 2*self.v_std) or (velocities[i]<self.v_mean - 2*self.v_std)
                omega_outlier = (omegas[i]>self.w_mean+2*self.w_std) or (omegas[i]<self.w_mean-2*self.w_std)
                
                start_idx = max(i-40, 0)
                end_idx = min(n_samples, i+40)

                if(velocity_outlier and omega_outlier):
                    if(v_w_idx == -1):
                        range_dict["v+w"].append([start_idx, end_idx])
                        v_w_idx+=1
                    else:
                        if(i <= range_dict["v+w"][v_w_idx][1]):
                            range_dict["v+w"][v_w_idx][1] = end_idx
                        elif(i-40 <= range_dict["v+w"][v_w_idx][1]):
                            range_dict["v+w"][v_w_idx][1] = end_idx
                        else:
                            range_dict["v+w"].append([start_idx, end_idx])
                            count_outliers += range_dict["v+w"][v_w_idx][1] - range_dict["v+w"][v_w_idx][0]
                            v_w_idx+=1

                elif(velocity_outlier):
                    if(v_idx == -1):
                        range_dict["v"].append([start_idx, end_idx])
                        v_idx+=1
                    else:
                        if(i <= range_dict["v"][v_idx][1]):
                            range_dict["v"][v_idx][1] = end_idx
                        elif(i-40 <= range_dict["v"][v_idx][1]):
                            range_dict["v"][v_idx][1] = end_idx
                        else:
                            range_dict["v"].append([start_idx, end_idx])
                            count_outliers += range_dict["v"][v_idx][1] - range_dict["v"][v_idx][0]
                            v_idx+=1

                elif(omega_outlier):
                    if(w_idx == -1):
                        range_dict["w"].append([start_idx, end_idx])
                        w_idx+=1
                    else:
                        if(i <= range_dict["w"][w_idx][1]):
                            range_dict["w"][w_idx][1] = end_idx
                        elif(i-40 <= range_dict["w"][w_idx][1]):
                            range_dict["w"][w_idx][1] = end_idx
                        else:
                            range_dict["w"].append([start_idx, end_idx])
                            count_outliers += range_dict["w"][w_idx][1] - range_dict["w"][w_idx][0]
                            w_idx+=1

        print(f"{scene} Outlier Count: {count_outliers}/{n_samples}")
        return range_dict
    
    def process_file(self, h5_filepath, scene, outlier_dict):
        with h5py.File(h5_filepath, "r") as h5file, h5py.File(self.output_h5_path_expert, "a") as output_h5_expert, h5py.File(self.output_h5_path_other, "a") as output_h5_other:

            positions = h5file["pos"][:]
            orientations = h5file["heading"][:]

            n_samples = h5file["image"].shape[0]
            print(f"Scene: {scene}, Samples: {n_samples}")

            self.last_action = np.array([h5file["v"][0][0], h5file["omega"][0][0]])

            v_w_outlier_idx = 0
            v_outlier_idx = 0
            w_outlier_idx = 0

            v_w_top, v_top, w_top = None, None, None
            v_w_bottom, v_bottom, w_bottom = None, None, None

            n_v_w_seg = len(outlier_dict["v+w"])
            n_v_seg = len(outlier_dict["v"])
            n_w_seg = len(outlier_dict["w"])

            if(n_v_w_seg>0):
                v_w_top = outlier_dict["v+w"][0][1]
                v_w_bottom = outlier_dict["v+w"][0][0]
            if(n_v_seg>0):
                v_top = outlier_dict["v"][0][1]
                v_bottom = outlier_dict["v"][0][0]
            if(n_w_seg>0):
                w_top = outlier_dict["w"][0][1]
                w_bottom = outlier_dict["w"][0][0]

            group_names = ["0", "1", "2", "3"]  # 0 : v+w outliers, 1 : v outliers, 2: w outliers, 3: normal
            
            for group in group_names:
                # print(group, group in output_h5)
                if group not in output_h5_expert:
                    output_h5_expert.create_group(group)
                    output_h5_other.create_group(group)

                    expert_datasets = {
                        key: self.create_or_get_dataset(
                            output_h5_expert[group],
                            key,
                            (0,) + h5file[value].shape[1:],
                            dtype=h5file[value].dtype
                        )
                        for key, value in {"image_t":"image", "image_next":"image"}.items()
                    }

                    other_datasets = {
                        key: self.create_or_get_dataset(
                            output_h5_other[group],
                            key,
                            (0,) + h5file[key].shape[1:],
                            dtype=h5file[key].dtype
                        )
                        for key in ["image"]
                    }

            for i in range(n_samples - 2 ): # Avoid last two samples as the next state will not have any 

                goal_distance, heading_error, goal_idx = self.calculate_goal_features(i, positions, orientations)
                next_state_goal_d , next_state_heading_e = self.calculate_next_state_goal_features(i+1, goal_idx, positions, orientations)

                if(self.verbose):
                    print(goal_idx, min(n_samples, i + self.v_max * 375), goal_distance, positions[i], positions[goal_idx])
                # print(h5file["v"][i:i+2])
                expert_action = (h5file["v"][i][0], h5file["omega"][i][0])
                L_v, L_w, expert_v_idx, expert_w_idx = self.create_action_space(self.last_action, expert_action)

                expert_action = np.array([L_v[expert_v_idx], L_w[expert_w_idx]])
                other_actions = []
                all_actions = [expert_action]
                for v_idx in range(len(L_v)):
                    for w_idx in range(len(L_w)):

                        if(v_idx == expert_v_idx and w_idx == expert_w_idx):
                            continue
                        else:
                            action = [L_v[v_idx], L_w[w_idx]]           # Action as (velocity, omega)
                            other_actions.append(action)     
                            all_actions.append(action)

                other_actions = np.array(other_actions)
                all_actions = np.array(all_actions)

                if(self.verbose):
                    print(f"Sample: {i} Last_action: {self.last_action} Expert_action: {expert_action} Goal Index: {goal_idx}")
                    print(other_actions)

                group, v_w_outlier_idx, v_outlier_idx, w_outlier_idx, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom = self.get_group(i, outlier_dict, v_w_top, v_w_bottom, 
                                                                                                                                                v_top, v_bottom, w_top, w_bottom, 
                                                                                                                                                n_v_w_seg, n_v_seg, n_w_seg,
                                                                                                                                                v_w_outlier_idx, v_outlier_idx, w_outlier_idx)
                
                v_t = h5file["v"][i][0]
                w_t = h5file["omega"][i][0]

                v_next = h5file["v"][i+1][0]
                w_next = h5file["omega"][i+1][0]

                image_t = h5file["image"][i]
                image_next = h5file["image"][i+1]

                rewards = self.extract_rewards(goal_distance, heading_error, v_t, w_t, self.last_action, all_actions, image_t)
                rewards = np.array(rewards.cpu())

                v_t = h5file["v"][i][0]
                w_t = h5file["omega"][i][0]

                # Append data to the selected group

                self.append_to_group_dataset(output_h5_expert[group], "image_t", image_t, (0,) + image_t.shape)
                self.append_to_group_dataset(output_h5_expert[group], "goal_distance_t", goal_distance, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "heading_error_t", heading_error, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "v_t", v_t, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "w_t", w_t, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "action", expert_action, (0, 2))
                self.append_to_group_dataset(output_h5_expert[group], "last_action", self.last_action, (0, 2))
                self.append_to_group_dataset(output_h5_expert[group], "reward", rewards[0][0], (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "image_next", image_next, (0,) + image_next.shape)
                self.append_to_group_dataset(output_h5_expert[group], "goal_distance_next", next_state_goal_d, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "heading_error_next", next_state_heading_e, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "v_t_next", v_next, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "w_t_next", w_next, (0, 1))

                other_rewards = rewards[0][1:]

                image_idx = len(output_h5_other[group]["image"])  # Get the index for this pair


                self.append_to_group_dataset(output_h5_other[group], "image", image_t, (0,) + image_t.shape)
                self.append_to_group_dataset(output_h5_other[group], "image", image_next, (0,) + image_next.shape)

                for j in range(24):
                    self.append_to_group_dataset(output_h5_other[group], "image_t_index", np.array([image_idx]), (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "goal_distance_t", goal_distance, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "heading_error_t", heading_error, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "v_t", v_t, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "w_t", w_t, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "action", other_actions[j], (0, 2))
                    self.append_to_group_dataset(output_h5_other[group], "last_action", self.last_action, (0, 2))
                    self.append_to_group_dataset(output_h5_other[group], "reward", other_rewards[j], (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "image_next_index", np.array([image_idx+1]), (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "goal_distance_next", next_state_goal_d, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "heading_error_next", next_state_heading_e, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "v_t_next", v_next, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "w_t_next", w_next, (0, 1))

                self.last_action = np.array(expert_action)
                                                                                       
    def get_group(self, i, outlier_dict, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom, n_v_w_seg, n_v_seg, n_w_seg, v_w_outlier_idx, v_outlier_idx, w_outlier_idx):
        # print(i, outlier_dict, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom, n_v_w_seg, n_v_seg, n_w_seg, v_w_outlier_idx, v_outlier_idx, w_outlier_idx)
        if(v_w_top):
            if(i > v_w_top):
                v_w_outlier_idx+=1
                if(v_w_outlier_idx < n_v_w_seg):
                    v_w_top, v_w_bottom = outlier_dict["v+w"][v_w_outlier_idx][1], outlier_dict["v+w"][v_w_outlier_idx][0]
        if(v_top):
            if(i > v_top):
                v_outlier_idx+=1
                if(v_outlier_idx < n_v_seg):
                    v_top, v_bottom = outlier_dict["v"][v_outlier_idx][1], outlier_dict["v"][v_outlier_idx][0]

        if(w_top):
            if(i > w_top):
                w_outlier_idx+=1
                if(w_outlier_idx < n_w_seg):
                    
                    w_top, w_bottom = outlier_dict["w"][w_outlier_idx][1], outlier_dict["w"][w_outlier_idx][0]
                    # print(i, w_outlier_idx, w_bottom, w_top)
        group_dict = {
            "v+w": "0",
            "v": "1",
            "w": "2",
        }

        if(v_w_bottom):
            if (i > v_w_bottom and i <= v_w_top):
                return group_dict["v+w"], v_w_outlier_idx, v_outlier_idx, w_outlier_idx, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom
        if(v_bottom):
            if (i > v_bottom and i <= v_top):
                return group_dict["v"], v_w_outlier_idx, v_outlier_idx, w_outlier_idx, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom
        if(w_bottom):  
            # print(i, w_bottom, w_top)
            if( i > w_bottom and i <= w_top):
                # print(i)
                return group_dict["w"], v_w_outlier_idx, v_outlier_idx, w_outlier_idx, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom
        
        return "3", v_w_outlier_idx, v_outlier_idx, w_outlier_idx, v_w_top, v_w_bottom, v_top, v_bottom, w_top, w_bottom

    def calculate_preferences(self, L_v, L_w, preferences, expert_v_idx, expert_w_idx):

        distance_v_array = [-abs(L_v[i] - L_v[expert_v_idx])/self.v_res for i in range(len(L_v))]
        distance_w_array = [-abs(L_w[i] - L_w[expert_w_idx])/self.w_res for i in range(len(L_w))]

        # distance_v_array = [-abs(i - expert_v_idx) for i in range(len(L_v))]
        # distance_w_array = [-abs(i - expert_w_idx) for i in range(len(L_w))]

        # print(-abs(0 - expert_v_idx), -abs(L_v[0] - L_v[expert_v_idx])/self.v_res, self.v_res)

        distance_v_array = np.array(distance_v_array)
        distance_w_array = np.array(distance_w_array)

        tau_v = self.create_tau_array(preferences[:2], L_v, expert_v_idx, True)
        tau_w = self.create_tau_array(preferences[2:], L_w, expert_w_idx, False)

        tau_v = np.array(tau_v)
        tau_w = np.array(tau_w)

        pref_score_v = np.exp(distance_v_array/tau_v).reshape((1,tau_v.shape[0]))
        pref_score_w = np.exp(distance_w_array/tau_w).reshape((1,tau_w.shape[0]))

        return pref_score_v, pref_score_w
    
    def create_tau_array(self, preferences, L, expert_idx, is_v):
        if(preferences[0] and preferences[1]):
            tau_arr = [self.tau_2 for i in range(len(L))]
        
        elif(not (preferences[0] or preferences[1])):
            tau_arr = [self.tau_1 for i in range(len(L))]
            tau_arr[expert_idx] = self.tau_2
        
        elif(preferences[0]):
            if(is_v):
                tau_arr = [self.tau_1 for i in range(expert_idx)] + [self.tau_2 for i in range(len(L)-expert_idx)]
            else:
                tau_arr = [self.tau_2 for i in range(expert_idx+1)] + [self.tau_1 for i in range(len(L)-expert_idx-1)]

        elif(preferences[1]):
            if(is_v):
                tau_arr = [self.tau_2 for i in range(expert_idx+1)] + [self.tau_1 for i in range(len(L)-expert_idx-1)]
            else:
                tau_arr = [self.tau_1 for i in range(expert_idx)] + [self.tau_2 for i in range(len(L)-expert_idx)]

        return tau_arr
    
    def preference_generator(self, pref_matrix, L_v, L_w):
        for j in range(self.n_v*self.n_w):
            for k in range(j+1, self.n_v*self.n_w):
                
                v_1_idx = j//self.n_w
                w_1_idx = j%self.n_v
                v_1, w_1 = (L_v[v_1_idx], L_w[w_1_idx])
                
                v_2_idx = k//self.n_w
                w_2_idx = k%self.n_v
                v_2, w_2 = (L_v[v_2_idx], L_w[w_2_idx])

                preference_value = (pref_matrix[v_1_idx][w_1_idx])/(pref_matrix[v_1_idx][w_1_idx] + pref_matrix[v_2_idx][w_2_idx])

                if(preference_value < 0.5):
                    pref_label = 0
                else:
                    pref_label = 1

                yield [v_1, w_1, v_2, w_2, preference_value, pref_label]

    def calculate_goal_features(self, current_idx, positions, orientations):
        """Calculate distance to goal and heading error."""
        if current_idx >= len(positions) - 1:
            return 0, 0  # Edge case for the last sample
        
        # goal_idx = np.random.randint(i + 1, min(n_samples, i + self.v_max * 375)) #250 is 25 Hz times 15 seoncds

        goal_idx = np.random.randint(current_idx + 1, min(len(positions), current_idx + self.v_max * 375))

        current_pos = positions[current_idx][:2]
        goal_pos = positions[goal_idx][:2]

        distance = np.linalg.norm(goal_pos - current_pos)

        desired_heading = orientations[goal_idx]
        current_yaw = orientations[current_idx]
        heading_error = normalize_angle(desired_heading - current_yaw)

        return distance, heading_error, goal_idx
    
    def calculate_next_state_goal_features(self, next_idx, goal_idx, positions, orientations):
        """Calculate distance to goal and heading error."""
        if next_idx >= len(positions) - 1:
            return 0, 0  # Edge case for the last sample
        
        # goal_idx = np.random.randint(i + 1, min(n_samples, i + self.v_max * 375)) #250 is 25 Hz times 15 seoncds

        current_pos = positions[next_idx][:2]
        goal_pos = positions[goal_idx][:2]

        distance = np.linalg.norm(goal_pos - current_pos)

        desired_heading = orientations[goal_idx]
        current_yaw = orientations[next_idx]
        heading_error = normalize_angle(desired_heading - current_yaw)

        return distance, heading_error

    def create_or_get_dataset(self, group, name, shape, dtype=np.float32):
        if name in group:
            return group[name]
        else:
            return group.create_dataset(name, shape=shape, maxshape=(None,) + shape[1:], dtype=dtype)

    def append_to_dataset(self, dataset, data):
        dataset.resize((dataset.shape[0] + 1), axis=0)
        dataset[-1] = data
    
    def append_to_group_dataset(self, group, dataset_name, data, shape ):
        """Append data to a dataset within a specific HDF5 group."""

        # print(group.keys(), dataset_name, shape)
        if dataset_name not in group:           
            dtype = data.dtype if isinstance(data, np.ndarray) else np.float32
            group.create_dataset(dataset_name, shape=shape,  maxshape = (None,) + shape[1:], dtype=dtype)

        dataset = group[dataset_name]
        dataset.resize((dataset.shape[0] + 1), axis=0)
        dataset[-1] = data

# Example Usage
h5_dir = "/media/gershom/Media/Datasets/SCAND/Annotated"
output_h5_path_expert = "/media/gershom/Media/Datasets/SCAND/scand_rl_data_grouped_expert.h5"
output_h5_path_other = "/media/gershom/Media/Datasets/SCAND/scand_rl_data_grouped_other.h5"
reward_model_path = "/media/gershom/Media/Datasets/SCAND/model_3_epoch_70.pth"
scand_stats_path = "/home/gershom/Documents/GAMMA/IROS25/Repos/Offline-IRL/src/data_stats.json"
# completed = ["A_Spot_Bass_Rec_Fri_Nov_26_126_annotated.h5", "A_Spot_Library_Fountain_Tue_Nov_9_35_annotated.h5", "A_Spot_Union_Union_Wed_Nov_10_67_annotated.h5",
#             "A_Spot_Union_Library_Tue_Nov_9_37_annotated.h5", "A_Spot_Stadium_PerformingArtsCenter_Sat_Nov_13_106_annotated.h5", "A_Spot_AHG_Library_Mon_Nov_8_24_annotated.h5",
#             "A_Spot_Ballstructure_UTTower_Wed_Nov_10_60_annotated.h5", "A_Spot_AHG_Library_Fri_Nov_5_21_annotated.h5", "A_Spot_Thompson_Butler_Sat_Nov_13_103_annotated.h5", 
#             "C_Spot_Speedway_Butler_Fri_Nov_26_131_annotated.h5", "A_Spot_AHG_GDC_Tue_Nov_9_41_annotated.h5", "D_Spot_PerformingArts_Lbj_Sat_Nov_13_98_annotated.h5",
#             "A_Spot_NHB_Ahg_Wed_Nov_10_55_annotated.h5", "A_Spot_Library_MLK_Thu_Nov_18_122_annotated.h5", "A_Spot_Library_MLK_Thu_Nov_18_123_annotated.h5",
#             "C_Spot_SanJac_Bass_Fri_Nov_26_125_annotated.h5", "B_Spot_AHG_Union_Mon_Nov_15_111_annotated.h5", "A_Spot_NHB_Jester_Wed_Nov_10_62_annotated.h5",
#             "A_Spot_UTTower_Union_Wed_Nov_10_61_annotated.h5", "A_Spot_Welch_Union_Thu_Nov_11_71_annotated.h5", "A_Spot_Ahg_EERC_Thu_Nov_11_79_annotated.h5",
#             "A_Spot_Fountain_Dobie_Fri_Nov_12_83_annotated.h5", "A_Spot_AHG_Cola_Sat_Nov_13_96_annotated.h5", "A_Spot_Security_NHB_Wed_Nov_10_54_annotated.h5",
#             "B_Spot_AHG_Library_Tue_Nov_9_34_annotated.h5", "A_Spot_Parlin_Parlin_Wed_Nov_10_51_annotated.h5", "C_Spot_Tent_AHG_Fri_Nov_26_130_annotated.h5",
#             "A_Spot_Library_Fountain_Mon_Nov_8_30_annotated.h5", "C_Spot_Rec_Tent_Fri_Nov_26_129_annotated.h5", "A_Spot_Ahg_Library_Wed_Nov_10_56_annotated.h5",
#             "A_Spot_AHG_Library_Wed_Nov_10_46_annotated.h5", "A_Jackal_Speedway_Speedway_Fri_Oct_29_12_annotated.h5"]

completed = []

processor = SCANDRLProcessor(output_h5_path_expert, output_h5_path_other, reward_model_path, scand_stats_path = scand_stats_path)

for filename in os.listdir(h5_dir):
    # print(filexit()ename)
    if filename.endswith(".h5"):
        if filename in completed:
            continue

        scene_name = "_".join(filename.split("_")[1:4])
        outlier_dict = processor.get_outliers(os.path.join(h5_dir, filename), scene_name)
        print(outlier_dict)
        processor.process_file(os.path.join(h5_dir, filename), scene_name, outlier_dict)    

        # raise Exception