from utils import RewardModelSCAND
import torch

model = RewardModelSCAND()

batch_size = 4
image = torch.randn(batch_size, 3, 224, 224)  # Random image input
goal_distance = torch.randn(batch_size, 1)
heading_error = torch.randn(batch_size, 1)
velocity = torch.randn(batch_size, 2)
past_action = torch.randn(batch_size, 2)
current_action = torch.randn(batch_size, 2)

# Forward pass
reward = model(image, goal_distance, heading_error, velocity, past_action, current_action)

print("Reward Output Shape:", reward.shape)
print("Reward Output:", reward)