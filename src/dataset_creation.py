import h5py
import os
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image

from utils.reward_model_scand_5 import RewardModelSCAND  # Ensure correct model class
from scipy.stats import truncnorm

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

        self.n_v = n_v
        self.n_w = n_w
        self.last_action = None  # Initialize this later for each scene
        self.tau_1 = 0.32
        self.tau_2 = 1

        self.output_h5_path_expert = output_h5_path_expert
        self.output_h5_path_other = output_h5_path_other
        self.verbose = False
        self.goal_verbose = True
        self.outlier_window = 40
        self.transform = False

        self.v_res = 0.3
        self.w_res = 0.24

        self.d_v_max = (self.n_w - 1) * self.v_res / 2
        self.d_w_max = (self.n_w - 1) * self.w_res / 2

        # Create consolidated HDF5 file if it doesn't exist
        if not os.path.exists(output_h5_path_expert):
            with h5py.File(output_h5_path_expert, "w") as f:
                pass
                
        if not os.path.exists(output_h5_path_other):
            with h5py.File(output_h5_path_other, "w") as f:
                pass
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model = RewardModelSCAND()

        checkpoint = torch.load(reward_model_path, map_location=self.device)
        self.reward_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.reward_model.to(self.device)
        self.reward_model.eval()

        self.means = {
            "goal_distance": 9.886707493584415,
            "heading_error": -0.00850201072417097,
            "velocity": 1.2765753737615684,
            "omega": -0.002504312331437613,
            "preference_ranking": np.array([ 1.2765752877383227, -0.0025043122945302026])
        }

        self.stds = {
            "goal_distance": 6.374522710777637,
            "heading_error": 0.44817681236970364,
            "velocity": 0.2818386933609228,
            "omega": 0.1492379970642606,
            "preference_ranking": np.array([ 0.2818391436539243, 0.14923799976918717])
        }

        # **DINOv2 Transformations (Resize + Normalize)**
        self.dino_transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            transforms.Resize((224, 224)),  # Ensure images are resized properly
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])

        self.end_of_scene = False
        self.goal_idx = -1
        self.dt = 1/25
        self.total_count = 0

        self.goal_scaling_factor = 2
        self.goal_reaching_bonus = 2


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
    
    def extract_social_rewards(self, current_action, image):
    
        current_action = self.standardize(current_action, "preference_ranking")  
        image = np.array(self.load_image(image))
        
        current_action = torch.from_numpy(np.array(current_action, dtype=np.float32)).to(self.device)
        image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0).to(self.device)

        # print(goal_distance.shape)
        
        with torch.no_grad():

            rewards = self.reward_model(image, current_action, 1)

        # print(current_action)
        # print(rewards)
        # raise Exception
        return rewards   

    def create_action_space(self, expert_action):

        # print(last_action)
        expert_v, expert_w = expert_action

        # Ensure velocities and omegas stay within bounds
        v_actions = np.clip(np.linspace(expert_v - 2*self.v_res, expert_v + 2*self.v_res, self.n_v), 0, self.v_max)
        w_actions = np.clip(np.linspace(expert_w - 2*self.w_res, expert_w + 2*self.w_res, self.n_w), -self.w_max, self.w_max)

        # Find expert indices
        expert_v_idx = self.n_v//2
        expert_w_idx = self.n_w//2

        v_actions[expert_v_idx] = expert_v
        w_actions[expert_w_idx] = expert_w
        
        return v_actions, w_actions, expert_v_idx, expert_w_idx
    
    def generate_action_pairs(self, v_actions, w_actions):
        # Create grid of all possible (v, w) pairs
        V, W = np.meshgrid(v_actions, w_actions, indexing='ij')  # shape (n_v, n_w)
        
        # Flatten the grids to list of pairs
        all_actions = np.stack([V.ravel(), W.ravel()], axis=1)  # shape (n_v * n_w, 2)
        return all_actions

    def get_outliers(self, h5_filepath, scene):

        group_list = {
            "v>wr": [],
            "v>wl": [],
            "v<wr": [],
            "v<wl":[],
            "wr": [],
            "wl": [],
            "v>": [],
            "v<": [],
            "r" : [],
            "l" : [],
        }
        count_outliers = 0
        with h5py.File(h5_filepath, "r") as h5file:

            velocities = h5file["v"][:]
            omegas = h5file["omega"][:]
            n_samples = h5file["image"].shape[0]

            v_w_idx = -1
            v_idx = -1
            w_idx = -1


            fast_outlier = velocities>self.v_mean + 2*self.v_std 
            slow_outlier = velocities<self.v_mean - 2*self.v_std

            right_outlier = omegas<self.w_mean-2*self.w_std
            left_outlier = omegas>self.w_mean+2*self.w_std

            right = omegas < 0

            left = omegas >= 0

            # raise Exception
            # group_list["v>wr"] = np.logical_and(fast_outlier, right_outlier)
            # group_list["v>wl"] = np.logical_and(fast_outlier, left_outlier)
            group_list["v<wr"] = np.logical_and(slow_outlier, right_outlier)
            group_list["v<wl"] = np.logical_and(slow_outlier, left_outlier)
            group_list["v>"] = fast_outlier
            group_list["v<"] = slow_outlier
            group_list["wr"] = right_outlier
            group_list["wl"] = left_outlier
            group_list["r"] = right
            group_list["l"] = left

        for key in group_list.keys():
            print(np.sum(group_list[key]))
            count_outliers += np.sum(group_list[key])

        print(f"{scene} Outlier Count: {count_outliers}/{n_samples}")
        return group_list
    
    def process_file(self, h5_filepath, scene, group_dict):
        with h5py.File(h5_filepath, "r") as h5file, h5py.File(self.output_h5_path_expert, "a") as output_h5_expert, h5py.File(self.output_h5_path_other, "a") as output_h5_other:

            positions = h5file["pos"][:]
            orientations = h5file["heading"][:]

            n_samples = h5file["image"].shape[0]
            print(f"Scene: {scene}, Samples: {n_samples}")

            self.last_action = np.array([h5file["v"][0][0], h5file["omega"][0][0]])

            group_names = ["2", "3", "4", "5", "6", "7", "8", "9"] #  "v<wr", "v<wl", "wr", "wl", "v>", "v<", "r" , "l" 
            
            
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
            
            i = 0
            self.end_of_scene = False
            if(n_samples == 0 ):
                raise Exception
            
            while not self.end_of_scene:
                if (i == n_samples-1):
                    self.end_of_scene = True
                    continue

                if( i > self.goal_idx):

                    if(i == 0):
                        pass
                    else:
                        i = self.calculate_start_idx(i, positions) 

                    self.goal_idx = self.calculate_goal_idx(i, n_samples-1, positions)

                    if (self.goal_verbose):
                        print(f"Start idx: {i}, Goal idx: {self.goal_idx}")

                # Gather relevant data
                group = self.get_group(i, group_dict)
                image_t = h5file["image"][i]
                image_next = h5file["image"][i+1]

                expert_action =np.array([h5file["v"][i][0], h5file["omega"][i][0]])
                L_v, L_w, expert_v_idx, expert_w_idx = self.create_action_space(expert_action)
                expert_action_idx = self.n_v * expert_v_idx + expert_w_idx
                # Generate Action pairs
                all_actions = self.generate_action_pairs(L_v, L_w)
                goal_distance, heading_error = self.calculate_goal_features(i, self.goal_idx, positions, orientations)
                next_goal_distance, next_heading_error = self.project_traj_batch(all_actions, goal_distance, heading_error)
                
                if(self.verbose):
                    print(f"Sample: {i} Last_action: {self.last_action}, Expert_action: {expert_action}, Goal Index: {self.goal_idx}, Goal distance: {goal_distance}")

                social_rewards = self.extract_social_rewards(all_actions, image_t)
                social_rewards = np.array(social_rewards.cpu())
                
                non_social_rewards = self.calc_rewards_non_social(goal_distance, next_goal_distance, next_heading_error, all_actions, expert_action)
                reward = social_rewards[0][expert_action_idx] + non_social_rewards[expert_action_idx]
                
                # Append data to the selected group
                self.append_to_group_dataset(output_h5_expert[group], "image_t", image_t, (0,) + image_t.shape)
                self.append_to_group_dataset(output_h5_expert[group], "goal_distance_t", goal_distance, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "heading_error_t", heading_error, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "action", expert_action, (0, 2))
                self.append_to_group_dataset(output_h5_expert[group], "reward", reward , (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "image_next", image_next, (0,) + image_next.shape)
                self.append_to_group_dataset(output_h5_expert[group], "goal_distance_next", next_goal_distance[expert_action_idx], (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "heading_error_next", next_heading_error[expert_action_idx], (0, 1))

                image_idx = len(output_h5_other[group]["image"])  # Get the index for this pair


                self.append_to_group_dataset(output_h5_other[group], "image", image_t, (0,) + image_t.shape)
                self.append_to_group_dataset(output_h5_other[group], "image", image_next, (0,) + image_next.shape)

                # Loop over 25 actions (except expert's)
                for j in range(len(all_actions)):
                    if j == expert_action_idx:
                        continue  # Skip expert action, already saved

                    other_action = all_actions[j]
                    reward = social_rewards[0][j] + non_social_rewards[j]

                    self.append_to_group_dataset(output_h5_other[group], "image_t_index", np.array([image_idx]), (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "goal_distance_t", goal_distance, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "heading_error_t", heading_error, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "action", other_action, (0, 2))
                    self.append_to_group_dataset(output_h5_other[group], "reward", reward, (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "image_next_index", np.array([image_idx+1]), (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "goal_distance_next", next_goal_distance[j], (0, 1))
                    self.append_to_group_dataset(output_h5_other[group], "heading_error_next", next_heading_error[j], (0, 1))

                self.last_action = np.array(expert_action)
                i+=1
                self.total_count+=1
            self.goal_idx = -1
    
    def project_traj_batch(self, actions, current_goal_distance, current_heading_error, epsilon=1e-6):
        """
        actions: numpy array of shape (N, 2) each row is [v, omega]
        Returns:
            distances: array of shape (N,)
            theta_errors: array of shape (N,)
        """
        T = self.dt
        v = actions[:, 0]
        omega = actions[:, 1]
        
        # Goal position in robot frame (scalar for all actions)
        x_goal = current_goal_distance * np.cos(current_heading_error)
        y_goal = current_goal_distance * np.sin(current_heading_error)

        # Avoid division by zero for small omega
        omega_safe = np.where(np.abs(omega) > epsilon, omega, np.sign(omega) * epsilon + (omega == 0)*epsilon)

        r = v / omega_safe

        sin_wt = np.sin(omega * T)
        cos_wt = np.cos(omega * T)

        # Compute robot motion
        x_robot = np.where(np.abs(omega) > epsilon, r * sin_wt, v * T)
        y_robot = np.where(np.abs(omega) > epsilon, r * (1 - cos_wt), np.zeros_like(v))
        theta_robot = omega * T

        # Translate goal relative to robot’s motion
        dx = x_goal - x_robot
        dy = y_goal - y_robot

        cos_theta = np.cos(theta_robot)
        sin_theta = np.sin(theta_robot)

        # Rotate goal into new robot frame
        x_goal_new = cos_theta * dx + sin_theta * dy
        y_goal_new = -sin_theta * dx + cos_theta * dy

        # Compute distance and heading error
        distances = np.sqrt(x_goal_new**2 + y_goal_new**2)
        theta_errors = np.arctan2(y_goal_new, x_goal_new)
        theta_errors = normalize_angle(theta_errors)

        return distances, theta_errors

    def calc_rewards_non_social(self, curr_distance_to_goal, next_distance_to_goal, next_heading_error, next_action, expert_action):

        r_progress = np.tanh(((curr_distance_to_goal - next_distance_to_goal)/(expert_action[0]*self.dt + 1e-6)) * self.goal_scaling_factor)

        r_heading = np.cos(next_heading_error)
        r_dynamic = -np.sum(np.abs(next_action - self.last_action)/[4*self.v_res, 4*self.w_res], axis = 1)
        r_goal = self.goal_reaching_bonus / (1 + np.exp(2 * (next_distance_to_goal)))


        return (r_goal + r_heading +  r_dynamic + r_progress)/4
                                                                                 
    def get_group(self, i, group_dict):
        group_keys = {
            # "v>wr": "0",
            # "v>wl": "1",
            "v<wr": "2",
            "v<wl": "3",
            "wr": "4",
            "wl": "5",
            "v>": "6",
            "v<": "7",
            "r": "8",
            "l": "9"
        }
        
        # if(group_dict["v>wr"][i]):
        #     return group_keys["v>wr"]
        
        # elif(group_dict["v>wl"][i]):
        #     return group_keys["v>wl"]
        
        # el
        if(group_dict["v<wr"][i]):
            return group_keys["v<wr"]
        
        elif(group_dict["v<wl"][i]):
            return group_keys["v<wl"]
        
        elif(group_dict["wr"][i]):
            return group_keys["wr"]
        
        elif(group_dict["wl"][i]):
            return group_keys["wl"]
        
        elif(group_dict["v>"][i]):
            return group_keys["v>"]
        
        elif(group_dict["v<"][i]):
            return group_keys["v<"]
        
        elif(group_dict["r"][i]):
            return group_keys["r"]
        
        elif(group_dict["l"][i]):
            return group_keys["l"]
        
        else:
            print("Error this should not have happened")
            return "10"

    def calculate_start_idx(self, current_idx, positions):
        
        overlap_distance = 2
        current_pos = positions[current_idx][:2]
        past_positions = positions[0:current_idx, :2]  # all positions before i

        if len(past_positions) == 0:
            return current_idx

        diffs = past_positions - current_pos  
        dists = np.linalg.norm(diffs, axis=1)

        valid_indices = np.where(dists <= overlap_distance)[0]

        if len(valid_indices) == 0:
            return current_idx

        # Randomly choose one valid index
        start_idx = np.random.choice(valid_indices)

        return start_idx
    
    def calculate_goal_idx(self, current_idx, n_samples, positions):
        """Calculate distance to goal and heading error."""

        mean_distance = 10.0  # meters
        std_distance = 5.0    # meters

        current_pos = positions[current_idx, :2]
        future_positions = positions[current_idx+1:n_samples, :2]

        diffs = future_positions - current_pos
        distance = np.linalg.norm(diffs, axis = 1)
        offsets = np.arange(1, len(distance) + 1)

        if np.max(distance) < mean_distance - std_distance:
            return n_samples

        # print("goal_stats", mean, sigma)
        lower_bound_dist = mean_distance - std_distance
        upper_bound_dist = mean_distance + std_distance

        # Truncated Normal Sampling
        a, b = (lower_bound_dist - mean_distance) / std_distance, (upper_bound_dist - mean_distance) / std_distance
        sampled_dist = truncnorm.rvs(a, b, loc=mean_distance, scale=std_distance)

        # Find closest index offset where distance ≈ sampled_dist
        closest_offset = offsets[np.argmin(np.abs(distance - sampled_dist))]
        goal_idx = current_idx + closest_offset

        return goal_idx
    
    def calculate_goal_features(self, current_idx, goal_idx, positions, orientations):
        current_pos = positions[current_idx][:2]
        goal_pos = positions[goal_idx][:2]

        distance = np.linalg.norm(goal_pos - current_pos)

        desired_heading = orientations[goal_idx]
        current_yaw = orientations[current_idx]
        heading_error = normalize_angle(desired_heading - current_yaw)

        return distance, heading_error

    def create_or_get_dataset(self, group, name, shape, dtype=np.float32):
        if name in group:
            return group[name]
        else:
            return group.create_dataset(name, shape=shape, maxshape=(None,) + shape[1:], dtype=dtype)
    
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
output_h5_path_expert = "/media/gershom/Media/Datasets/SCAND/scand_rl_data_grouped_expert_train.h5"
output_h5_path_other = "/media/gershom/Media/Datasets/SCAND/scand_rl_data_grouped_other_train.h5"
reward_model_path = "/media/gershom/Media/HALO/Models/model_4_epoch_16.pth"
scand_stats_path = "/home/gershom/Documents/GAMMA/IROS25/Repos/Offline-IRL/src/data_stats.json"


completed = [
            # "A_Spot_Bass_Rec_Fri_Nov_26_126_annotated.h5", "A_Spot_Library_Fountain_Tue_Nov_9_35_annotated.h5", "A_Spot_Union_Union_Wed_Nov_10_67_annotated.h5",
            # "A_Spot_Union_Library_Tue_Nov_9_37_annotated.h5", "A_Spot_Stadium_PerformingArtsCenter_Sat_Nov_13_106_annotated.h5", "A_Spot_AHG_Library_Mon_Nov_8_24_annotated.h5",
            # "A_Spot_Ballstructure_UTTower_Wed_Nov_10_60_annotated.h5", "A_Spot_AHG_Library_Fri_Nov_5_21_annotated.h5", "A_Spot_Thompson_Butler_Sat_Nov_13_103_annotated.h5", 
            # "C_Spot_Speedway_Butler_Fri_Nov_26_131_annotated.h5", "A_Spot_AHG_GDC_Tue_Nov_9_41_annotated.h5", "D_Spot_PerformingArts_Lbj_Sat_Nov_13_98_annotated.h5",
            # "A_Spot_NHB_Ahg_Wed_Nov_10_55_annotated.h5", "A_Spot_Library_MLK_Thu_Nov_18_122_annotated.h5", "A_Spot_Library_MLK_Thu_Nov_18_123_annotated.h5",
            # "C_Spot_SanJac_Bass_Fri_Nov_26_125_annotated.h5", "B_Spot_AHG_Union_Mon_Nov_15_111_annotated.h5", "A_Spot_NHB_Jester_Wed_Nov_10_62_annotated.h5",
            # "A_Spot_UTTower_Union_Wed_Nov_10_61_annotated.h5", "A_Spot_Welch_Union_Thu_Nov_11_71_annotated.h5", "A_Spot_Ahg_EERC_Thu_Nov_11_79_annotated.h5",
            # "A_Spot_Fountain_Dobie_Fri_Nov_12_83_annotated.h5", "A_Spot_AHG_Cola_Sat_Nov_13_96_annotated.h5", "A_Spot_Security_NHB_Wed_Nov_10_54_annotated.h5",
            # "B_Spot_AHG_Library_Tue_Nov_9_34_annotated.h5", "A_Spot_AHG_AHG_Mon_Nov_8_27_annotated.h5", "A_Spot_SAC_GeorgeWashStatue_Wed_Nov_10_50_annotated.h5" 
            
            "A_Spot_Parlin_Parlin_Wed_Nov_10_51_annotated.h5", "C_Spot_Tent_AHG_Fri_Nov_26_130_annotated.h5",
            "A_Spot_Library_Fountain_Mon_Nov_8_30_annotated.h5", "C_Spot_Rec_Tent_Fri_Nov_26_129_annotated.h5", "A_Spot_Ahg_Library_Wed_Nov_10_56_annotated.h5",
            "A_Spot_AHG_Library_Wed_Nov_10_46_annotated.h5", "A_Jackal_Speedway_Speedway_Fri_Oct_29_12_annotated.h5"
]

# completed = []

processor = SCANDRLProcessor(output_h5_path_expert, output_h5_path_other, reward_model_path, scand_stats_path = scand_stats_path)

for filename in os.listdir(h5_dir):
    # print(filexit()ename)
    if filename.endswith(".h5"):
        if filename in completed:
            continue

        scene_name = "_".join(filename.split("_")[1:4])
        group_dict = processor.get_outliers(os.path.join(h5_dir, filename), scene_name)
        # print(group_dict)
        processor.process_file(os.path.join(h5_dir, filename), scene_name, group_dict)    

        # raise Exception