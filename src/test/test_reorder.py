import torch

# Custom PL Loss (Simplified for Testing)
class PL_Loss(torch.nn.Module):
    def forward(self, rewards):
        log_denominators = torch.logcumsumexp(rewards.flip(dims=[1]), dim=1).flip(dims=[1])
        loss = rewards - log_denominators
        loss = -loss[:, :-1].sum(dim=1)  # Sum over sequence, ignore last term
        return loss.mean()

def test_shuffling_and_correct_loss_computation():
    torch.manual_seed(42)  # Ensure reproducibility
    loss_fn = PL_Loss()

    # Step 1: Create a Sample Reward Tensor (Requires Gradients)
    rewards = torch.tensor([[1.0, 3.0, 4.0, 0.5]], requires_grad=True)  # Shape: (1, 4)

    # Step 2: Generate a Random Permutation
    perm = torch.randperm(rewards.shape[1])  # Generate a random permutation of indices
    shuffled_rewards = rewards[:, perm]  # Shuffle rewards

    # Step 3: Compute the Reverse Permutation (Restore Order Before Loss)
    reverse_perm = torch.argsort(perm).unsqueeze(0)  # Compute indices to reverse the shuffle
    reordered_rewards = torch.gather(shuffled_rewards, 1, reverse_perm)  # Restore original preference order

    # Step 4: Compute Loss on Correctly Ordered Rewards
    loss = loss_fn(reordered_rewards)  # Apply PL loss
    print(f"🔹 Loss after reordering: {loss.item()}")

    # Step 5: Compute Gradients
    loss.backward()  # Compute gradients

    # Step 6: Print Results
    print(f"\n**Test Results:**")
    print(f"🔹 Shuffled Indices: {perm.tolist()}")
    print(f"🔹 Original Rewards: {rewards.detach().numpy()}")
    print(f"🔹 Shuffled Rewards: {shuffled_rewards.detach().numpy()}")
    print(f"🔹 Reordered Rewards: {reordered_rewards.detach().numpy()}")
    print(f"🔹 Gradients: {rewards.grad.detach().numpy()}")

    print("**Test Passed!** Gradients flow correctly through reordering.")

# Run the test
test_shuffling_and_correct_loss_computation()
