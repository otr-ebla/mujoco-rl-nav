# assets/custompolicy.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from typing import Tuple

class LidarTransformer(nn.Module):
    def __init__(self, n_stacking: int, num_rays: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(num_rays, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, n_stacking * num_rays]
        B = x.size(0)
        x = x.view(B, -1, self.input_proj.in_features)  # [B, n_stacking, num_rays]
        x = self.input_proj(x)                          # [B, n_stacking, d_model]
        x = self.encoder(x)                             # [B, n_stacking, d_model]
        x = x.mean(dim=1)                               # [B, d_model]
        return x

class CustomTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_stacking: int, num_rays: int, output_dim: int = 256):
        super().__init__(observation_space, features_dim=output_dim)
        self.n_stacking = n_stacking
        self.num_rays = num_rays
        self.lidar_size = n_stacking * num_rays
        self.polar_size = observation_space.shape[0] - self.lidar_size

        self.transformer = LidarTransformer(n_stacking, num_rays, d_model=128)
        self.polar_mlp = nn.Sequential(
            nn.Linear(self.polar_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.combined = nn.Sequential(
            nn.Linear(128 + 64, output_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lidar = obs[:, :self.lidar_size]
        polar = obs[:, self.lidar_size:]
        lidar_features = self.transformer(lidar)
        polar_features = self.polar_mlp(polar)
        combined = torch.cat((lidar_features, polar_features), dim=1)
        return self.combined(combined)
