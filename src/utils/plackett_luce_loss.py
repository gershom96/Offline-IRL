import torch
import torch.nn as nn

# Define Plackett-Luce loss function
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
        log_denominators = torch.logcumsumexp(rewards, dim=1)
        loss = rewards - log_denominators
        loss = -loss[:, :-1].sum(dim=1)

        return loss.mean()  # Average over batch
