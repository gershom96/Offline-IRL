import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import Dinov2Model

# Positional Encoding Class
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

# Reward Model
class RewardModelSCAND(nn.Module):
    def __init__(self, num_queries=4, hidden_dim = 768):  # Multi-query support
        super().__init__()

        self.hidden_dim = hidden_dim  # 768 is DINOv2 feature size
        self.num_queries = num_queries  # Number of state queries

        # Load DINOv2
        self.vision_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.vision_dim = self.hidden_dim

        # Freeze DINOv2 weights
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(self.hidden_dim)

        # LayerNorm for Patch Embeddings (Before and After Self-Attention)
        self.patch_norm = nn.LayerNorm(self.hidden_dim)  

        # Self-Attention Over Vision Features
        self.attn_layer = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # MLP for numerical inputs (goal distance, heading error, velocity, past and current action)
        self.state_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_dim)
        )
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Multi-Query Learnable Queries
        self.state_query_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_queries)
        ])

        # Cross-Attention and Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=8, 
            batch_first=True  # Ensures input is (batch_size, seq_len, hidden_dim)
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)  # Normalize after fusion

        # **MLP-based Query Fusion**
        self.query_fusion_mlp = nn.Sequential(
            nn.Linear(self.num_queries * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),  # Output fused representation
            nn.LayerNorm(self.hidden_dim)  # Normalize fused representation
        )

        # Reward Prediction Head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),  # Replace ReLU with GELU
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Normalize output range
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Zero-bias initialization

    def forward(self, image, goal_distance, heading_error, velocity, omega, last_action, preference_ranking, batch_size):
        """
        image: (batch_size, 3, 224, 224)
        goal_distance: (batch_size, 1)
        heading_error: (batch_size, 1)
        velocity: (batch_size, 1)
        omega: (batch_size, 1)
        last_action: (batch_size, 2)
        preference_ranking: (batch_size, 25, 2)  # 25 ranked action pairs
        """
        
        # Extract vision features (Patch embeddings, excluding CLS)
        patch_embeddings = self.vision_model(image).last_hidden_state[:, 1:, :]  # Shape: (batch_size, num_patches, hidden_dim)
        patch_embeddings = self.positional_encoding(patch_embeddings)  # Shape: (batch_size, num_patches, hidden_dim)
        patch_embeddings = self.patch_norm(patch_embeddings)  # Normalizing before patch embeddings

        # Self-Attention on Vision Features
        attn_output, _ = self.attn_layer(patch_embeddings, patch_embeddings, patch_embeddings)  # Shape: (batch_size, num_patches, hidden_dim)
        attn_output = self.attn_norm(attn_output)  # Normalize After Self-Attention
        # Add Positional Encoding Again Before Cross-Attention
        attn_output = self.positional_encoding(attn_output)  # Add Positional Encoding Again Before Cross-Attention

        # Process State Inputs
        state_inputs = torch.cat([goal_distance, heading_error, velocity, omega, last_action, preference_ranking], dim=-1)  # (batch_size, 25, 8)
        state_embedding = self.state_mlp(state_inputs)  # Shape: (batch_size, 25, hidden_dim)
        state_embedding = self.state_norm(state_embedding) # Norm to normalize before fusion

        # Generate Multiple Queries
        query_list = [proj(state_embedding) for proj in self.state_query_proj]
        state_queries = torch.stack(query_list, dim=2)  # (batch_size, 25, Q, hidden_dim)


        # Reshape to match MultiheadAttention expected shape (batch_size, seq_length, hidden_dim)
        state_queries = state_queries.view(batch_size * 25, self.num_queries, -1)  # (batch_size * 25, num_queries, hidden_dim)
# Ensure attn_output is expanded properly
        attn_output = attn_output.unsqueeze(1).expand(-1, 25, -1, -1)  # Shape: (batch_size, 25, num_patches, hidden_dim)
        # print(attn_output.shape)
        # Reshape correctly before cross-attention
        attn_output = attn_output.reshape(batch_size * 25, attn_output.shape[2], attn_output.shape[3])  # Shape: (batch_size * 25, num_patches, hidden_dim)
        # print(attn_output.shape)
        # Cross-Attention (Querying vision features with action-specific queries)
        fused_features, _ = self.cross_attention(state_queries, attn_output, attn_output, key_padding_mask=None)  # (batch_size * 25, num_queries, hidden_dim)

        # print("Average Attention Weights:", _.mean(dim=-1))
        # Reshape fused features back to (batch_size, 25, num_queries, hidden_dim)
        fused_features = fused_features.view(batch_size, 25, self.num_queries, -1)

        # print(fused_features.shape)
        fused_features = fused_features.reshape(batch_size, 25, -1)  # (batch_size, 25, hidden_dim)
        # **MLP-based Query Fusion**
        fused_features = fused_features.view(batch_size, 25, -1) # Shape: (batch_size, 25, num_queries * hidden_dim)
        fused_features = self.query_fusion_mlp(fused_features)   # Shape: (batch_size, 25, hidden_dim)

        fused_features = self.fusion_norm(fused_features)  # Normalize After Feature Fusion

        # Predict rewards for all 25 actions
        rewards = self.reward_head(fused_features).squeeze(-1)  # (batch_size, 25)

        return rewards