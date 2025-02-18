import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plackett_luce_loss_v2 import PL_Loss as PL_v2
from utils.plackett_luce_loss import PL_Loss as PL_v1


# Unit test for PL_Loss with random permutation to check gradient flow
def test_pl_loss_with_gradient_check():
    loss1 = PL_v1()
    loss2 = PL_v2()

    gamma = torch.tensor([])
    rewards_2 = torch.tensor([[4.0, 3.0, 2.0, 5.0]], requires_grad=True)  
    perm = torch.tensor([[3, 0, 1, 2 ]])

    rewards_1 = torch.tensor([[5.0, 4.0, 3.0, 2.0]], requires_grad=True)  

    
    loss_1 = loss1(rewards_1)
    loss_2 = loss2(rewards_2, perm)

    # 🚀 Step 3: Compute losses
    loss_1.backward()
    loss_2.backward()
  
    
    print(f"Loss 1: {loss_1}")
    print(f"Loss 2: {loss_2}")

    # 🚀 Step 5: Print gradients to verify they flow correctly
    print("Gradients for loss 1:", rewards_1.grad)
    print("Gradients for loss 2:", rewards_2.grad)


# Run the test
test_pl_loss_with_gradient_check()
