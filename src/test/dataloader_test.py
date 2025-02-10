import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import PreferenceDataset

# Paths
h5_file = "/media/gershom/Media/Datasets/NuScenes/H5/nuscenes_preference_data.h5"
nuscenes_dataset_path = "/media/gershom/Media/Datasets/NuScenes"

# Initialize Dataset
mode = 2  # Change this to test different camera modes
dataset = PreferenceDataset(h5_file, nuscenes_dataset_path, mode=mode, time_window=1)
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

# Display Images and Metadata
for i in range(n):
    sample = dataset[i]

    print(f"Sample : {i}/{n}")
    # print(sample["goal_distance"].shape)
    print("Goal Distance:", sample["goal_distance"][0])
    print("Heading Error:", sample["heading_error"][0])
    print("Velocity:", sample["velocity"][0].numpy())
    print("Rotation Rate:", sample["rotation_rate"][0].numpy())
    print("Preference Ranking Shape:", sample["preference_ranking"].shape)

    # ---- Display the Images ----
    images = sample["images"][0]  # Shape: [#Cameras, 3, H, W]
    num_cameras = images.shape[0]

    print("No. Cameras:", num_cameras)

    # Adjust grid based on the number of cameras
    cols = min(3, num_cameras)
    rows = (num_cameras + cols - 1) // cols  # Ceiling division

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten() if num_cameras > 1 else [axs]  # Ensure axs is iterable

    for idx, img_tensor in enumerate(images):
        img = img_tensor.permute(1, 2, 0).numpy().astype('uint8')  # Convert back to [H, W, C]

        axs[position_labels[mode][idx]].imshow(img)
        axs[position_labels[mode][idx]].axis('off')

        # Dynamic title based on camera mode
        if mode == 1:
            title = camera_labels[0]  # Front Camera
        elif mode == 2:
            title = camera_labels[mode_labels[mode][idx]]  # Front & Back
        elif mode == 3:
            title = camera_labels[mode_labels[mode][idx]]  # Front, Back, Side
        elif mode == 4:
            title = camera_labels[mode_labels[mode][idx]]  # All 6 Cameras
        else:
            title = f"Camera {idx + 1}"

        axs[idx].set_title(title)

    plt.tight_layout()
    plt.show()

    # Break after first few samples for testing
    # if i == 2:
    #     break



# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # Converts to [0,1] float and rearranges to [C, H, W]
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# dataset = PreferenceDataset("path_to_h5.h5", transform=transform)