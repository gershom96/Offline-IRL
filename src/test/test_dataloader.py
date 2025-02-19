import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.nuscenes_pref_dataset import NuScenesPreferenceDataset
from data.scand_pref_dataset import SCANDPreferenceDataset
from data.scand_pref_dataset_2 import SCANDPreferenceDataset2
from data.scand_pref_dataset_3 import SCANDPreferenceDataset3

scand = True
# Paths

if(scand):
    h5_file =  "/media/gershom/Media/Datasets/SCAND/scand_preference_data_grouped.h5"
else:
    h5_file = "/media/gershom/Media/Datasets/NuScenes/H5/nuscenes_preference_data.h5"
    nuscenes_dataset_path = "/media/gershom/Media/Datasets/NuScenes"
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # Converts to [0,1] float and rearranges to [C, H, W]
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# dataset = PreferenceDataset("path_to_h5.h5", transform=transform)


if(scand):
    dataset = SCANDPreferenceDataset3(h5_file)
    n = dataset.length
    camera_labels = [
            'CAM_FRONT']
else:

    # Initialize Dataset
    mode = 1
    dataset = NuScenesPreferenceDataset(h5_file, nuscenes_dataset_path, mode=mode, time_window=1)
    n = len(dataset)

        # Camera Labels for Reference
    camera_labels = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT',  'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT','CAM_BACK_RIGHT', 
        ]

    mode_labels = {
                1: [0],            # Front Camera
                2: [0, 3],         # Front & Back
                3: [0, 3, 4, 5],   # Front, Back, Side-Back
                4: [ 2, 0, 1 , 4, 3, 5]  # All 6 Cameras
            }

    position_labels = {
                1: [0],            # Front Camera
                2: [0, 1],         # Front & Back
                3: [1, 4, 3, 5],   # Front, Back, Side-Back
                4: [1, 2, 0 , 4, 3, 5]  # All 6 Cameras
            }


fig = plt.figure(figsize=(12, 8))
plt.ion()  # interactive mode on
current_num_cameras = None
axs = None

# Display Images and Metadata
for i in range(n):
    sample = dataset[i]

    print(f"Group: {dataset.indices_to_group[i]}")
    print(f"Sample : {i}/{n}")
    print(sample["goal_distance"].shape)
    print("Goal Distance:", sample["goal_distance"][0][0])
    print("Heading Error:", sample["heading_error"][0][0])
    print("Velocity:", sample["velocity"][0][0].numpy())
    print("Rotation Rate:", sample["rotation_rate"][0][0].numpy())
    print("Preference Ranking Shape:", sample["preference_ranking"].shape)
    print("Preference Indices Shape:", sample["pref_idx"].shape)
    print("Preference Ranking Top3:", sample["preference_ranking"][0], sample["preference_ranking"][1], sample["preference_ranking"][2])
    print("Last Action: ", sample["last_action"][0][0])
    print("Last Action: ", sample["last_action"][0][1])

    # ---- Display the Images ----
    images = sample["images"]  # Shape: [#Cameras, 3, H, W]
    num_cameras = images.shape[0]
    print("No. Cameras:", num_cameras)

    # Create (or update) subplots only if number of cameras has changed.
    if current_num_cameras != num_cameras:
        fig.clf()  # clear the existing figure
        cols = min(3, num_cameras)
        rows = (num_cameras + cols - 1) // cols  # Ceiling division
        axs = fig.subplots(rows, cols)
        # Ensure axs is a flat list.
        if num_cameras > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
        current_num_cameras = num_cameras

    # Now update the images in the existing axes.
    for idx, img_tensor in enumerate([images]):
        print(img_tensor.shape)
        img = img_tensor.permute(1, 2, 0).numpy().astype('uint8')  # [H, W, 3]
        if not scand:
            # For nuScenes, use the mapping (if available).
            if mode in [2, 3, 4]:
                ax_idx = position_labels[mode][idx]
            else:
                ax_idx = idx  # default to index order
            if hasattr(axs[ax_idx], "im_obj"):
                axs[ax_idx].im_obj.set_data(img)
            else:
                axs[ax_idx].im_obj = axs[ax_idx].imshow(img)
            axs[ax_idx].axis("off")
            axs[ax_idx].set_title(camera_labels[mode_labels[mode][idx]] if mode in [2,3,4] else camera_labels[0])
        else:
            # For SCAND, assume a single camera.
            if hasattr(axs[0], "im_obj"):
                axs[0].im_obj.set_data(img)
            else:
                axs[0].im_obj = axs[0].imshow(img)
            axs[0].axis("off")
            axs[0].set_title(camera_labels[0])
    
    fig.suptitle(f"Sample {i+1}/{n}", fontsize=14)
    plt.tight_layout()
    fig.canvas.draw_idle()

    print("Press any key in the figure window to proceed to the next sample...")
    plt.waitforbuttonpress()  # Wait for key press without closing the window.

# Turn off interactive mode when done.
plt.ioff()
