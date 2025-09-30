import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

class CNNLSTMExtractor(BaseFeaturesExtractor):
    """
    Turns a flat stacked vector into (B, T, 1, L) with T=time_steps and L=obs_dim//T,
    then applies Conv1D -> LSTM and returns a 128-D feature vector.
    """
    def __init__(self, observation_space, time_steps=10, conv_channels=(16, 32), lstm_hidden=128):
        obs_dim = int(observation_space.shape[0])
        assert obs_dim % time_steps == 0, \
            f"Obs dim {obs_dim} must be divisible by time_steps {time_steps}"
        super().__init__(observation_space, features_dim=lstm_hidden)
        self.time_steps = time_steps
        self.length = obs_dim // time_steps

        self.conv1 = nn.Conv1d(1,  conv_channels[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2)
        self.fc_in = conv_channels[1] * self.length
        self.lstm = nn.LSTM(self.fc_in, lstm_hidden, batch_first=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        x = obs.view(B, self.time_steps, 1, self.length)              # (B,T,1,L)
        x = x.reshape(B * self.time_steps, 1, self.length)            # (B*T,1,L)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(B, self.time_steps, -1)                          # (B,T,features)
        x, _ = self.lstm(x)                                           # (B,T,hidden)
        return x[:, -1, :]                                            # last step (B,hidden)
