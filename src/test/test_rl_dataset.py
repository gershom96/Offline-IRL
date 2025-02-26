import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms


# Paths to your created datasets
output_h5_path_expert = "/media/gershom/Media/Datasets/SCAND/test/scand_rl_data_grouped_expert.h5"
output_h5_path_other = "/media/gershom/Media/Datasets/SCAND/test/scand_rl_data_grouped_other.h5"

def check_dataset_consistency(h5_path, check_images=True, num_samples=5):
    """Verify dataset integrity and alignment."""

    dino_transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
        transforms.Resize((224, 224)),  # Ensure images are resized properly
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
    ])

    with h5py.File(h5_path, "r") as h5file:
        print(f"Checking dataset: {h5_path}")

        # Print available groups
        groups = list(h5file.keys())
        print(f"Groups found: {groups}")

        for group in groups:
            print(f"\n Checking group: {group}")

            # Check if required datasets exist
            required_keys = ["goal_distance_t", "heading_error_t", "v_t", "w_t", "action", "reward"]
            missing_keys = [key for key in required_keys if key not in h5file[group]]
            
            if missing_keys:
                print(f"⚠️ Missing keys in {group}: {missing_keys}")
                continue
            
            # Read some random samples
            total_samples = h5file[group]["goal_distance_t"].shape[0]
            if total_samples == 0:
                print(f"⚠️ No samples in {group}.")
                continue

            sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
            print(f"Found {total_samples} samples. Checking {len(sample_indices)} random samples...")

            for idx in range(total_samples):
                # Read data
                goal_distance = h5file[group]["goal_distance_t"][idx]
                heading_error = h5file[group]["heading_error_t"][idx]
                velocity = h5file[group]["v_t"][idx]
                omega = h5file[group]["w_t"][idx]
                action = h5file[group]["action"][idx]
                reward = h5file[group]["reward"][idx]

                print(f"Sample {idx}: Goal Dist: {goal_distance}, Heading Err: {heading_error}, v: {velocity}, w: {omega}, Action: {action}, Reward: {reward}")

                # Check Image Alignment (Optional)
                if check_images and "image_t" in h5file[group] and "image_next" in h5file[group]:
                    img_t = h5file[group]["image_t"][idx]
                    img_next = h5file[group]["image_next"][idx]

                elif "image_t_index" in h5file[group] and "image_next_index" in h5file[group]:
                        # For the other dataset, retrieve images using indices
                        img_idx_t = h5file[group]["image_t_index"][idx][0]
                        img_idx_next = h5file[group]["image_next_index"][idx][0]

                        # Retrieve from /images dataset
                        img_t = h5file[group]["image"][img_idx_t]
                        img_next = h5file[group]["image"][img_idx_next]
                else:
                    print(" No images found in this group.")
                    continue
                

                if isinstance(img_t, np.ndarray):
                    image_t_bytes = img_t.tobytes()
                    image_next_bytes = img_next.tobytes()
                elif isinstance(img_t, (bytes, bytearray)):
                    image_t_bytes = bytes(img_t)
                    image_next_bytes = bytes(img_next)
                else:
                    raise ValueError("Unsupported type for image_data: {}".format(type(image_data)))

                stream_t = BytesIO(image_t_bytes)
                image_t = Image.open(stream_t).convert("RGB")  # Decode the image and ensure 3 channels.

                image_t= transforms.ToTensor()(image_t) 
                image_t = (image_t.permute(1, 2, 0).numpy()*255).astype('uint8')

                stream_nxt = BytesIO(image_next_bytes)
                image_nxt = Image.open(stream_nxt).convert("RGB")  # Decode the image and ensure 3 channels.
                image_nxt= transforms.ToTensor()(image_nxt) 

                
                image_nxt = (image_nxt.permute(1, 2, 0).numpy()*255).astype('uint8')

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(image_t)
                axes[0].set_title(f"Image @ t (idx {idx})")
                axes[1].imshow(image_nxt)
                axes[1].set_title(f"Image @ t+1 (idx {idx})")
                plt.show()
# Run tests for both datasets
# check_dataset_consistency(output_h5_path_expert, check_images=True)
check_dataset_consistency(output_h5_path_other, check_images=True)