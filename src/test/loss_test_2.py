import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plackett_luce_loss import PL_Loss as PL_v1


import torch

class PL_Loss_v2(torch.nn.Module):
    def forward(self, rewards, pref_idx):
        """
        Args:
            rewards: Tensor of shape (batch_size, num_actions) with unordered predicted rewards.
            pref_idx: Tensor of shape (batch_size, num_actions), contains indices that define the correct ranking.
        
        Returns:
            Scalar loss value.
        """
        # Reorder rewards based on preference indices
        
        # print(rewards.shape, pref_idx.dtype)
        ordered_rewards = torch.gather(rewards, 1, pref_idx)  # Align with true preference ranking

        # Compute PL Loss
        log_denominators = torch.logcumsumexp(ordered_rewards.flip(dims=[1]), dim=1).flip(dims=[1])
        loss = ordered_rewards - log_denominators
        loss = -loss[:, :-1].sum(dim=1)  # Sum over sequence, ignore last term

        return loss.mean()  # Average over batch


# Unit test for PL_Loss with random permutation to check gradient flow
def test_pl_loss_with_gradient_check(n=5):
    loss1 = PL_v1()
    loss2 = PL_Loss_v2()

    rewards_1 = torch.rand(1,25, requires_grad=True) 
    # Sort only the first n elements
    sorted_part, _ = torch.sort(rewards_1, dim=1, descending=True)
    sorted_rewards = torch.cat((sorted_part[:, :n], rewards_1[:, n:]), dim=1)  # Merge sorted part with the rest

    perm = torch.argsort(torch.rand(1, 25), dim=1) # this is after the perm has been argsorted
    rewards_2 = torch.gather(sorted_rewards, 1, perm)

    perm_ = torch.argsort(perm, dim=1)


    # print(ordered_rewards)
    
    loss_1 = loss1(sorted_rewards)
    loss_2 = loss2(rewards_2, perm_)

    # 🚀 Step 3: Compute losses
    loss_1.backward(retain_graph=True)
    loss_2.backward(retain_graph=True)
  
    
    print(f"Loss 1: {loss_1}")
    print(f"Loss 2: {loss_2}")

    # 🚀 Step 5: Print gradients to verify they flow correctly
    print("Gradients for loss 1:", rewards_1.grad)
    print("Gradients for loss 2:", rewards_2.grad)


# Run the test
test_pl_loss_with_gradient_check(n=25)


# import os
# import sys
# import torch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils.plackett_luce_loss_v2 import PL_Loss as PL_v2
# from utils.plackett_luce_loss import PL_Loss as PL_v1


# # Unit test for PL_Loss with random permutation to check gradient flow
# def test_pl_loss_with_gradient_check():
#     loss1 = PL_v1()
#     loss2 = PL_v2()

#     gamma = torch.tensor([])
#     rewards_2 = torch.tensor([[4.0, 3.0, 2.0, 5.0]], requires_grad=True)  
#     perm = torch.tensor([[3, 0, 1, 2 ]])

#     rewards_1 = torch.tensor([[5.0, 4.0, 3.0, 2.0]], requires_grad=True)  

    
#     loss_1 = loss1(rewards_1)
#     loss_2 = loss2(rewards_2, perm)

#     # 🚀 Step 3: Compute losses
#     loss_1.backward()
#     loss_2.backward()
  
    
#     print(f"Loss 1: {loss_1}")
#     print(f"Loss 2: {loss_2}")

#     # 🚀 Step 5: Print gradients to verify they flow correctly
#     print("Gradients for loss 1:", rewards_1.grad)
#     print("Gradients for loss 2:", rewards_2.grad)


# # Run the test
# test_pl_loss_with_gradient_check()

