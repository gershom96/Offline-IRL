import torch

checkpoint_path = "/fs/nexus-scratch/gershom/IROS25/Offline-IRL/src/models/checkpoints/model_epoch_40.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("🔍 Keys in Checkpoint:")
for key in checkpoint.keys():
    # print(f"- {key}")
    try:
        print(checkpoint[key].keys(), key)
    except:
        print(key)
    # for key in checkpoint[key]:
    #     print(key)