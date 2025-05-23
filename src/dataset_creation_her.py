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
    def __init__(self, output_h5_path_expert, v_max=2, w_max=1.5, d_v_max=0.05, d_w_max=0.03, n_v=5, n_w=5, scand_stats_path = '../scand_data_stats.json'):
        
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
        self.verbose = False
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
                

        self.means = {
            "goal_distance": 9.886707493584415,
            "heading_error": -0.00850201072417097,
            "velocity": 1.2765753737615684,
            "omega": -0.002504312331437613,
            "last_action": np.array([1.28372591, -0.00149611]),  # (2,)
            "preference_ranking": np.array([ 1.2765752877383227, -0.0025043122945302026])
        }

        self.stds = {
            "goal_distance": 6.374522710777637,
            "heading_error": 0.44817681236970364,
            "velocity": 0.2818386933609228,
            "omega": 0.1492379970642606,
            "preference_ranking": np.array([ 0.2818391436539243, 0.14923799976918717])
        }

    def standardize(self, data, key):
        """Standardizes numerical values using precomputed mean and std."""
        if(key == "diff_action"):
            return (data - self.means["preference_ranking"] + self.means["last_action"])/np.sqrt( self.stds["preference_ranking"]**2 + self.stds["last_action"]**2+ 1e-8)
        else:
            return (data - self.means[key]) / (self.stds[key] + 1e-8)  # Avoid division by zero
    
    def extract_rewards(self, goal_distance, heading_error, velocity, omega, last_action, expert_action, lidarscan):
        
        goal_distance = self.standardize(goal_distance, "goal_distance")
        heading_error = self.standardize(heading_error, "heading_error")
        velocity = self.standardize(velocity, "velocity")
        omega = self.standardize(omega, "omega")

        diff_action = self.standardize(expert_action-last_action, "diff_action")

        last_action = self.standardize(last_action, "last_action")

        expert_action = self.standardize(expert_action, "preference_ranking")  

        reward = -5* np.log(np.exp(goal_distance)+1) - heading_error**2 - np.sum(diff_action**2) - 2*np.exp(min(lidarscan)/6.0)

        if(np.isnan(reward)):
            raise Exception
        return reward   
    
    def create_action_space(self, expert_action):

        # print(last_action)
        expert_v, expert_w = expert_action

        # Ensure velocities and omegas stay within bounds
        v_actions = np.clip(np.linspace(expert_v - 2*self.v_res, expert_v + 2*self.v_res, self.n_v), 0, self.v_max)
        w_actions = np.clip(np.linspace(expert_w - 2*self.w_res, expert_w + 2*self.w_res, self.n_w), -self.w_max, self.w_max)

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

        with h5py.File(h5_filepath, "r") as h5file:

            velocities = h5file["v"][:]
            omegas = h5file["omega"][:]
            n_samples = h5file["image"].shape[0]

            v_w_idx = -1
            v_idx = -1
            w_idx = -1

            count_outliers = 0

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
        with h5py.File(h5_filepath, "r") as h5file, h5py.File(self.output_h5_path_expert, "a") as output_h5_expert:

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

                    expert_datasets = {
                        key: self.create_or_get_dataset(
                            output_h5_expert[group],
                            key,
                            (0,) + h5file[value].shape[1:],
                            dtype=h5file[value].dtype
                        )
                        for key, value in {"image_t":"image", "image_next":"image"}.items()
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
                        i = self.calculate_start_idx(i) 

                    goal_idx = self.calculate_goal_idx(i, n_samples-1)
                    self.goal_idx = goal_idx

                goal_distance, heading_error = self.calculate_goal_features(i, self.goal_idx, positions, orientations)
                next_state_goal_d , next_state_heading_e = self.calculate_next_state_goal_features(i+1, self.goal_idx, positions, orientations)

                if(self.verbose):
                    print(self.goal_idx, goal_distance, positions[i], positions[goal_idx])

                expert_action =np.array([h5file["v"][i][0], h5file["omega"][i][0]])
                L_v, L_w, expert_v_idx, expert_w_idx = self.create_action_space(expert_action)


                if(self.verbose):
                    print(f"Sample: {i} Last_action: {self.last_action} Expert_action: {expert_action} Goal Index: {goal_idx}")
                    print(expert_action)

                group= self.get_group(i, group_dict)
                
                v_t = h5file["v"][i][0]
                w_t = h5file["omega"][i][0]

                v_next = h5file["v"][i+1][0]
                w_next = h5file["omega"][i+1][0]

                image_t = h5file["image"][i]
                image_next = h5file["image"][i+1]

                lidarscan = h5file["scan"][i]

                reward = self.extract_rewards(goal_distance, heading_error, v_t, w_t, self.last_action, expert_action, lidarscan)
                # print(f"Goal distance: {goal_distance}, Heading Error: {heading_error}, Min obst: {min(lidarscan)}, Reward: {reward}")
                v_t = h5file["v"][i][0]
                w_t = h5file["omega"][i][0]

                # Append data to the selected group

                self.append_to_group_dataset(output_h5_expert[group], "image_t", image_t, (0,) + image_t.shape)
                self.append_to_group_dataset(output_h5_expert[group], "goal_distance_t", goal_distance, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "heading_error_t", heading_error, (0, 1))
                # self.append_to_group_dataset(output_h5_expert[group], "v_t", v_t, (0, 1))
                # self.append_to_group_dataset(output_h5_expert[group], "w_t", w_t, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "action", expert_action, (0, 2))
                # self.append_to_group_dataset(output_h5_expert[group], "last_action", self.last_action, (0, 2))
                self.append_to_group_dataset(output_h5_expert[group], "reward", reward , (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "image_next", image_next, (0,) + image_next.shape)
                self.append_to_group_dataset(output_h5_expert[group], "goal_distance_next", next_state_goal_d, (0, 1))
                self.append_to_group_dataset(output_h5_expert[group], "heading_error_next", next_state_heading_e, (0, 1))
                # self.append_to_group_dataset(output_h5_expert[group], "v_t_next", v_next, (0, 1))
                # self.append_to_group_dataset(output_h5_expert[group], "w_t_next", w_next, (0, 1))


                self.last_action = np.array(expert_action)
                i+=1
                self.total_count+=1
            self.goal_idx = -1

    def project_traj(self, action, current_goal_distance, current_heading_error, N = 5):

        dT = self.dt/N
        x,y, theta = 0, 0, 0

        x_goal = current_goal_distance*np.cos(current_heading_error)
        y_goal = current_goal_distance*np.sin(current_heading_error)

        for i in range(N):
            x += action[0] * np.cos(theta) * dT
            y += action[0] * np.sin(theta) * dT
            theta += action[1] * dT

        distance = np.linalg.norm([x - x_goal, y - y_goal])
        theta_goal = np.arctan2(y_goal - y, x_goal - x)

        theta_err = theta_goal - theta

        theta_err = normalize_angle(theta_err)

        return distance, theta_err

    def calc_rewards_non_social(self, curr_distance_to_goal, next_distance_to_goal, next_heading_error, next_action):

        r_goal = (curr_distance_to_goal - next_distance_to_goal)/(self.means["velocity"]*self.dt) 
        r_heading = np.cos(next_heading_error)
        r_dynamic = -np.sum(np.abs(next_action - self.last_action))

        if(next_distance_to_goal<0.1):
            r_goal +=10

        return r_goal + r_heading +  r_dynamic
                                                                                 
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

    def calculate_start_idx(self, i):

        start_idx = max(0, np.random.randint(i - self.v_mean*40, i))

        return start_idx
    
    def calculate_goal_idx(self, current_idx, n_samples):
        """Calculate distance to goal and heading error."""
        # if current_idx >= len(positions) - 1:
        #     return 0, 0, None  # Edge case for the last sample

        mean = min(current_idx + self.v_mean * 210, n_samples)
        sigma = self.v_mean * 62.5

        # print("goal_stats", mean, sigma)
        lower_bound = max(mean - sigma, current_idx)
        upper_bound = min(mean + sigma, n_samples)

        # Truncated Normal Sampling
        a, b = (lower_bound - mean) / sigma, (upper_bound - mean) / sigma
        goal_idx = int(np.round(truncnorm.rvs(a, b, loc=mean, scale=sigma)))
        
        if(upper_bound- current_idx < 252):
        
            return n_samples

        return goal_idx
    
    def calculate_goal_features(self, current_idx, goal_idx, positions, orientations):
        current_pos = positions[current_idx][:2]
        goal_pos = positions[goal_idx][:2]

        distance = np.linalg.norm(goal_pos - current_pos)

        desired_heading = orientations[goal_idx]
        current_yaw = orientations[current_idx]
        heading_error = normalize_angle(desired_heading - current_yaw)

        return distance, heading_error
    
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
h5_dir = "/media/gershom/Media/Datasets/SCAND/Comparisons/H5/"
output_h5_path_expert = "/media/gershom/Media/Datasets/SCAND/scand_rl_data_grouped_expert_her_train.h5"

scand_stats_path = "/home/gershom/Documents/GAMMA/IROS25/Repos/Offline-IRL/src/data_stats.json"

completed = [
    'A_Spot_Security_NHB_Wed_Nov_10_54_w_laserscan.h5', 'C_Spot_Tent_AHG_Fri_Nov_26_130_w_laserscan.h5', 'B_Spot_AHG_Library_Mon_Nov_15_107_w_laserscan.h5', 
    'A_Spot_Ahg_Library_Wed_Nov_10_56_w_laserscan.h5', 'C_Spot_Rec_Tent_Fri_Nov_26_129_w_laserscan.h5', 'A_Spot_NHB_Ahg_Wed_Nov_10_55_w_laserscan.h5', 
    # 'D_Spot_PerformingArts_Lbj_Sat_Nov_13_98_w_laserscan.h5', 'A_Spot_Ahg_EERC_Thu_Nov_11_79_w_laserscan.h5', 'A_Spot_AHG_AHG_Mon_Nov_8_27_w_laserscan.h5', 
    # 'A_Spot_Union_Union_Wed_Nov_10_53_w_laserscan.h5', 'A_Spot_Union_Union_Wed_Nov_10_67_w_laserscan.h5', 'A_Spot_SAC_GeorgeWashStatue_Wed_Nov_10_50_w_laserscan.h5', 
    # 'C_Spot_Speedway_Butler_Fri_Nov_26_131_w_laserscan.h5', 'A_Spot_Ballstructure_UTTower_Wed_Nov_10_60_w_laserscan.h5', 'A_Spot_Library_Fountain_Mon_Nov_8_30_w_laserscan.h5', 
    # 'C_Spot_SanJac_Bass_Fri_Nov_26_125_w_laserscan.h5', 'A_Spot_Library_Fountain_Tue_Nov_9_35_w_laserscan.h5', 'A_Spot_AHG_Library_Fri_Nov_12_81_w_laserscan.h5', 
    # 'A_Spot_Parlin_Parlin_Wed_Nov_10_51_w_laserscan.h5', 'B_Spot_AHG_Union_Mon_Nov_15_111_w_laserscan.h5', 'A_Spot_Library_MLK_Thu_Nov_18_123_w_laserscan.h5', 
    # 'A_Jackal_Speedway_Speedway_Fri_Oct_29_12_w_laserscan.h5', 'A_Spot_AHG_Library_Mon_Nov_8_24_w_laserscan.h5', 'A_Spot_NHB_Jester_Wed_Nov_10_62_w_laserscan.h5', 
    # 'A_Spot_Library_MLK_Thu_Nov_18_122_w_laserscan.h5', 'A_Spot_Stadium_PerformingArtsCenter_Sat_Nov_13_106_w_laserscan.h5', 'A_Spot_Union_Library_Tue_Nov_9_37_w_laserscan.h5', 
    # 'A_Spot_AHG_Library_Fri_Nov_5_21_w_laserscan.h5', 'A_Spot_AHG_Cola_Sat_Nov_13_96_w_laserscan.h5', 'A_Spot_Fountain_Dobie_Fri_Nov_12_83_w_laserscan.h5', 
    # 'A_Spot_Welch_Union_Thu_Nov_11_71_w_laserscan.h5', 'A_Spot_Bass_Rec_Fri_Nov_26_126_w_laserscan.h5', 'A_Spot_AHG_Library_Wed_Nov_10_46_w_laserscan.h5', 
    # 'A_Spot_AHG_GDC_Tue_Nov_9_41_w_laserscan.h5', 'B_Spot_AHG_Library_Tue_Nov_9_34_w_laserscan.h5', 'A_Spot_UTTower_Union_Wed_Nov_10_61_w_laserscan.h5', 
    # 'A_Spot_Thompson_Butler_Sat_Nov_13_103_w_laserscan.h5'
    
    ]


processor = SCANDRLProcessor(output_h5_path_expert, scand_stats_path = scand_stats_path)

for filename in os.listdir(h5_dir):
    # print(filexit()ename)
    if filename.endswith(".h5"):
        if filename in completed:
            continue

        scene_name = "_".join(filename.split("_")[1:4])
        outlier_dict = processor.get_outliers(os.path.join(h5_dir, filename), scene_name)
        # print(outlier_dict)
        processor.process_file(os.path.join(h5_dir, filename), scene_name, outlier_dict)    

        # raise Exception