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

class QueryBasedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, max_patches=256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # Learnable query vector
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        # Sinusoidal Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim, max_patches)

    def forward(self, patch_embeddings):
        """
        patch_embeddings: (batch_size, num_patches, hidden_dim)
        Returns: (batch_size, hidden_dim) - Dynamically pooled feature vector
        """
        # Add positional encoding to patches
        patch_embeddings = self.positional_encoding(patch_embeddings)

        batch_size = patch_embeddings.shape[0]
        q = self.query.expand(batch_size, -1, -1)  # Expand query to batch size
        attn_output, _ = self.attn(q, patch_embeddings, patch_embeddings)  # Self-attention
        return attn_output.squeeze(1)  # Remove query dimension

class RewardModel(nn.Module):
    def __init__(self, use_dinov2=True):
        super(RewardModel, self).__init__()

        self.use_dinov2 = use_dinov2
        self.hidden_dim = 768  # DINOv2 feature size

        # Load DINOv2 (Pretrained)
        if use_dinov2:
            self.vision_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self.vision_dim = self.hidden_dim
        else:
            self.vision_model = None
            self.vision_dim = 0  # No image features if not using vision

        # Self-Attention Over Patch Embeddings
        self.attn_layer = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)

        # Query-Based Attention Pooling (with Positional Encoding)
        self.attention_pooling = QueryBasedAttentionPooling(self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

        # MLP for numerical inputs (goal distance, heading error, velocity, past and current action)
        self.state_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim)  # No ReLU here
        )
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Cross-Attention for Fusion
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
        if self.use_dinov2:
            patch_embeddings = self.vision_model(image).last_hidden_state[:, 1:, :]  # Shape: (batch_size, num_patches, hidden_dim)
        else:
            patch_embeddings = torch.zeros((batch_size, 0, self.hidden_dim), device=goal_distance.device)

        # Apply Self-Attention on Patch Features
        patch_embeddings = self.positional_encoding(patch_embeddings)
        attn_output, _ = self.attn_layer(patch_embeddings, patch_embeddings, patch_embeddings)  # Shape: (batch_size, num_patches, hidden_dim)

        # Apply Query-Based Attention Pooling (which now includes Positional Encoding)
        # vision_features = self.attention_pooling(attn_output)  # Shape: (batch_size, hidden_dim)
        # vision_features = self.norm(vision_features)  # Normalize after attention

        # Process numerical inputs
        state_inputs = torch.cat([goal_distance, heading_error, velocity, past_action, current_action], dim=-1)  # (batch_size, 8)
        state_embedding = self.state_mlp(state_inputs)  # Shape: (batch_size, 256)
        state_embedding = self.state_norm(state_embedding) # Norm to normalize before fusion
        state_embedding = state_embedding.unsqueeze(1)

        fused_features, _ = self.cross_attention(state_embedding, attn_output, attn_output) # Shape: (batch_size, 1, hidden_dim)
        fused_features = fused_features.squeeze(1)
        # Predict reward
        reward = self.reward_head(fused_features)  # (batch_size, 1)

        return reward