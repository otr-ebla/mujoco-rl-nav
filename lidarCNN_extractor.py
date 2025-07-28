import torch
import torch.nn as nn

class LidarCNNExtractor(nn.Module):
    """
    Input layout:
      • stacked_distances: (n_stacking,)
      • stacked_angles:    (n_stacking,)
      • lidar_stack:       (n_stacking * num_rays,)
    Output: 128‑D feature vector
    """
    def __init__(self, observation_space, n_stacking: int, num_rays: int):
        super().__init__()
        self.n_stacking = n_stacking
        self.num_rays = num_rays

        self.dist_idx     = slice(0, n_stacking)
        self.angle_idx    = slice(n_stacking, 2 * n_stacking)
        self.lidar_idx    = slice(2 * n_stacking, 2 * n_stacking + n_stacking * num_rays)

        # CNN over lidar stack
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_stacking, num_rays)
            conv_out = self.cnn(dummy).shape[1]

        # FC for distances
        self.dist_fc = nn.Sequential(
            nn.Linear(n_stacking, 64),
            nn.ReLU()
        )

        # FC for angles
        self.angle_fc = nn.Sequential(
            nn.Linear(n_stacking, 64),
            nn.ReLU()
        )

        # Final fusion
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(conv_out + 64 + 64, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU()
        )
        self.features_dim = 128

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = obs[..., self.dist_idx]     # (B, n_stacking)
        angle = obs[..., self.angle_idx]   # (B, n_stacking)
        lidar = obs[..., self.lidar_idx]   # (B, n_stacking * num_rays)

        lidar = lidar.view(-1, 1, self.n_stacking, self.num_rays)

        dist_feat = self.dist_fc(dist)     # (B, 64)
        angle_feat = self.angle_fc(angle)  # (B, 64)
        lidar_feat = self.cnn(lidar)       # (B, conv_out)

        combined = torch.cat([lidar_feat, dist_feat, angle_feat], dim=-1)
        return self.fc(combined)           # (B, 128)
