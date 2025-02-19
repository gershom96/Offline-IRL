import torch

class PL_Loss(torch.nn.Module):
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
        ordered_rewards = torch.gather(rewards, 1, pref_idx[:,:,0])  # Align with true preference ranking

        # Compute PL Loss
        log_denominators = torch.logcumsumexp(ordered_rewards.flip(dims=[1]), dim=1).flip(dims=[1])
        loss = ordered_rewards - log_denominators
        loss = -loss[:, :-1].sum(dim=1)  # Sum over sequence, ignore last term

        return loss.mean()  # Average over batch
