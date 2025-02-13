import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import RewardModelSCAND
import torch

model = RewardModelSCAND()

batch_size = 4
image = torch.randn(batch_size, 3, 224, 224)  # Random image input
goal_distance = torch.randn(batch_size, 1)
heading_error = torch.randn(batch_size, 1)
velocity = torch.randn(batch_size, 1)
omega = torch.randn(batch_size, 1)
past_action = torch.randn(batch_size, 2)
current_action = torch.randn(batch_size, 2)

for i in range(10):
    # Forward pass
    t1 = time.time()
    reward = model(image, goal_distance, heading_error, velocity, omega, past_action, current_action, batch_size)
    t2 = time.time()

    print(t2-t1)
print("Reward Output Shape:", reward.shape)
print("Reward Output:", reward)