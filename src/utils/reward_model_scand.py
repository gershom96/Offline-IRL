import torch
import torch.nn as nn
from transformers import Dinov2Model

# May not use this approach. Can have a variant and train the Query based attn pooling
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        """
        d_model: Hidden dimension size (e.g., 768 for DINOv2)
        max_len: Maximum number of patches (e.g., 256)
        """
        super().__init__()

        # Create a matrix for positional encodings (max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cos for odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, num_patches, d_model)
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)

class RewardModelSCAND(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_dim = 768  # DINOv2 feature size

        # Load DINOv2 (Pretrained)
        self.vision_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.vision_dim = self.hidden_dim
        
        # Positional Encoding for Vision Patches
        self.positional_encoding = SinusoidalPositionalEncoding(self.hidden_dim)

        # LayerNorm for Patch Embeddings (Before and After Self-Attention)
        self.patch_norm = nn.LayerNorm(self.hidden_dim)  
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # Self-Attention Over Patch Embeddings
        self.attn_layer = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)

        # MLP for numerical inputs (goal distance, heading error, velocity, past and current action)
        self.state_mlp = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim)  # No ReLU here
        )
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Cross-Attention for Fusion (State Queries Vision Features)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_dim)  # Normalize fused features

        # Reward Prediction Head
        self.reward_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: Reward Score
        )

    def forward(self, image, goal_distance, heading_error, velocity, past_action, current_action, batch_size):
        """
        image: (batch_size, 3, 224, 224)
        goal_distance: (batch_size, 1)
        heading_error: (batch_size, 1)
        velocity: (batch_size, 2)
        past_action: (batch_size, 2)
        current_action: (batch_size, 2)
        """

        # Extract vision features (Patch embeddings, excluding CLS)
        patch_embeddings = self.vision_model(image).last_hidden_state[:, 1:, :]  # Shape: (batch_size, num_patches, hidden_dim)
        
        # Apply Self-Attention on Patch Features
        patch_embeddings = self.positional_encoding(patch_embeddings)  # Shape: (batch_size, num_patches, hidden_dim)
        patch_embeddings = self.patch_norm(patch_embeddings)  # Normalizing before patch embeddings

        attn_output, _ = self.attn_layer(patch_embeddings, patch_embeddings, patch_embeddings)  # Shape: (batch_size, num_patches, hidden_dim)
        attn_output = self.attn_norm(attn_output)  # Normalize After Self-Attentio

        # Process numerical inputs
        state_inputs = torch.cat([goal_distance, heading_error, velocity, past_action, current_action], dim=-1)  # (batch_size, 7)
        state_embedding = self.state_mlp(state_inputs)  # Shape: (batch_size, hidden_dim)
        state_embedding = self.state_norm(state_embedding) # Norm to normalize before fusion
        state_embedding = state_embedding.unsqueeze(1) # Shape: (batch_size, 1,  hidden_dim)

        fused_features, _ = self.cross_attention(state_embedding, attn_output, attn_output) # Shape: (batch_size, 1, hidden_dim)
        fused_features = fused_features.squeeze(1) # Shape: (batch_size, hidden_dim)
        fused_features = self.fusion_norm(fused_features) # Normalize After Feature Fusion
        # Predict reward
        reward = self.reward_head(fused_features)  # (batch_size, 1)

        return reward