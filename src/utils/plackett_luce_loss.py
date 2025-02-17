import torch
import torch.nn as nn

class PL_Loss(nn.Module):
    def __init__(self):
        super(PL_Loss, self).__init__()

    def forward(self, rewards):
        """
        Args:
            rewards: Tensor of shape (batch_size, num_actions), where actions are already ranked.

        Returns:
            Scalar loss value.
        """
        log_denominators = torch.logcumsumexp(rewards.flip(dims=[1]), dim=1).flip(dims=[1])

        # print(f":rewards: {rewards} | denom: {log_denominators}")
        # Compute PL loss
        loss = rewards - log_denominators 
        loss = -loss[:, :-1].sum(dim=1)  # Sum over sequence, ignore last term

        return loss.mean()  # Average over batch
