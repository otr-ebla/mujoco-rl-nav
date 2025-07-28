import torch
import torch.nn as nn

class LidarTransformerExtractor(nn.Module):
    """
    Replaces CNN with Transformer for temporal modeling of lidar frames.
    Input:
      • lidar stack: (B, n_stacking, num_rays)
      • polar stack: (B, n_stacking * 2)
    Output:
      • 128‑D feature vector
    """
    def __init__(self, observation_space, n_stacking: int, num_rays: int):
        super().__init__()
        self.n_stacking = n_stacking
        self.num_rays = num_rays

        lidar_dim = n_stacking * num_rays
        polar_dim = n_stacking * 2
        total_dim = lidar_dim + polar_dim

        assert observation_space.shape[0] == total_dim, \
            f"Expected obs dim {total_dim}, got {observation_space.shape[0]}"

        self.lidar_idx = slice(0, lidar_dim)
        self.polar_idx = slice(lidar_dim, lidar_dim + polar_dim)

        self.embedding = nn.Linear(num_rays, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.polar_fc = nn.Sequential(
            nn.Linear(polar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.features_dim = 128

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lidar = obs[..., self.lidar_idx]  # shape: (B, lidar_dim)
        polar = obs[..., self.polar_idx]  # shape: (B, polar_dim)

        B = obs.size(0)
        lidar_seq = lidar.view(B, self.n_stacking, self.num_rays)  # (B, n_stacking, num_rays)

        x = self.embedding(lidar_seq)       # (B, n_stacking, 64)
        x = self.transformer(x)             # (B, n_stacking, 64)
        lidar_feat = x.mean(dim=1)          # (B, 64) → mean pooling

        polar_feat = self.polar_fc(polar)   # (B, 64)
        polar_feat *= 2.0

        combined = torch.cat([lidar_feat, polar_feat], dim=-1)  # (B, 128)
        return self.fc(combined)           # (B, 128)
