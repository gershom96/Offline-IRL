import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor

class RewardModel(nn.Module):
    def __init__(self, use_dinov2=True):
        super(RewardModel, self).__init__()

        self.use_dinov2 = use_dinov2

        # Load DINOv2 (Pretrained)
        if use_dinov2:
            self.vision_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self.vision_dim = 768
        else:
            self.vision_model = None
            self.vision_dim = 0  # No image features if not using vision

        # Cross-Attention Fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        # MLP for numerical inputs
        self.state_mlp = nn.Sequential(
            nn.Linear(6, 128),  # (Goal Distance, Heading Error, Velocity, Past Action)
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # MLP for final prediction
        self.reward_head = nn.Sequential(
            nn.Linear(256 + self.vision_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: Reward Score
        )

    def forward(self, image, goal_distance, heading_error, velocity, past_action):
        """
        image: (batch_size, 3, 224, 224)
        goal_distance: (batch_size, 1)
        heading_error: (batch_size, 1)
        velocity: (batch_size, 2)
        past_action: (batch_size, 2)
        """

        batch_size = goal_distance.shape[0]

        # Process vision features
        if self.use_dinov2:
            vision_features = self.vision_model(image).last_hidden_state[:, 0, :]  # CLS Token (batch_size, 768)
        else:
            vision_features = torch.zeros((batch_size, 0), device=goal_distance.device)  # If no vision

        # Process numerical inputs
        state_inputs = torch.cat([goal_distance, heading_error, velocity, past_action], dim=-1)  # (batch_size, 6)
        state_embedding = self.state_mlp(state_inputs)  # (batch_size, 256)

        # Apply Cross-Attention Fusion
        fused_features, _ = self.cross_attention(state_embedding.unsqueeze(1), vision_features.unsqueeze(1), vision_features.unsqueeze(1))
        fused_features = fused_features.squeeze(1)  # (batch_size, 256)

        # Concatenate fused features and predict reward
        combined_features = torch.cat([fused_features, vision_features], dim=-1)
        reward = self.reward_head(combined_features)  # (batch_size, 1)

        return reward