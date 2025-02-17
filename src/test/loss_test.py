import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plackett_luce_loss import PL_Loss

# Unit test for PL_Loss
def test_pl_loss():
    loss_fn = PL_Loss()


    # Define two different rankings of the same rewards
    rewards_better = torch.tensor([[4.0, 3.0, 2.0, 1.0]]).unsqueeze(-1)  # Best ranking (higher first)
    rewards_worse = torch.tensor([[3.0, 2.0, 4.0, 1.0]]).unsqueeze(-1)   # Poorly ordered ranking

    # Compute losses
    loss_better = loss_fn(rewards_better)
    loss_worse = loss_fn(rewards_worse)

    print(f"Loss (Better Ranking [4,3,2,1]): {loss_better.item()}")
    print(f"Loss (Worse Ranking [3,2,1,4]): {loss_worse.item()}")

    # Assert that better ranking has a lower loss
    assert loss_better < loss_worse, "Better ranking should have a lower loss than worse ranking"

    print("Test passed! ✅")

# Run the test
test_pl_loss()