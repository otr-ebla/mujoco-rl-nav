import os
import gym
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
import pickle

# === Configuration ===
input_dim = 112  # adjust to your input
hidden_dims = [256, 256]
output_dim = 2
batch_size = 128
lr = 3e-4
epochs = 1000
patience = 10
save_dir = "bc_policy_mlp"
os.makedirs(save_dir, exist_ok=True)

# === Dummy data loading (replace with real .npz loading) ===
data = np.load("expert_data/expert_data.npz")
obs = data["obs"]
act = data["actions"]

obs_mean = obs.mean(axis=0)
obs_std = obs.std(axis=0) + 1e-8
obs = (obs - obs_mean) / obs_std

# === Split ===
split = int(0.9 * len(obs))
train_dataset = TensorDataset(torch.tensor(obs[:split], dtype=torch.float32), torch.tensor(act[:split], dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(obs[split:], dtype=torch.float32), torch.tensor(act[split:], dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Model ===
class MLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = MLPPolicy(input_dim, hidden_dims, output_dim).to("cuda")
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float("inf")
no_improve = 0
train_losses, val_losses, epochs_list = [], [], []

try:
    for epoch in range(1, epochs + 1):
        model.train()
        train_mse_accum = 0.0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mse_accum += loss.item()
        avg_train = train_mse_accum / len(train_loader)

        model.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                val_mse_accum += loss_fn(pred, y).item()
        avg_val = val_mse_accum / len(val_loader)

        print(f"Epoch {epoch:03d}: Train MSE = {avg_train:.6f}, Val MSE = {avg_val:.6f}")

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'obs_mean': obs_mean,
                'obs_std': obs_std
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        epochs_list.append(epoch)

except KeyboardInterrupt:
    print("Interrupted. Saving current model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'obs_mean': obs_mean,
        'obs_std': obs_std
    }, os.path.join(save_dir, "interrupted_model.pt"))

# === SB3 export compatibility ===
print("Creating SB3 compatible model...")

# Dummy environment
class DummyEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(input_dim,), dtype=np.float32)
act_space = gym.spaces.Box(low=-1, high=1, shape=(output_dim,), dtype=np.float32)

# Reconstruct model
checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"))
model.load_state_dict(checkpoint['model_state_dict'])

# Create SB3 dummy env
dummy_env = DummyEnv(obs_space, act_space)
vec_norm_env = VecNormalize(DummyVecEnv([lambda: dummy_env]), norm_obs=True, norm_reward=False)
vec_norm_env.training = False

# Build SB3 PPO model and assign weights
policy_kwargs = dict(net_arch=[dict(pi=hidden_dims, vf=hidden_dims)])
sb3_model = PPO(
    policy=ActorCriticPolicy,
    env=vec_norm_env,
    policy_kwargs=policy_kwargs,
    device="cpu"
)
sb3_model.policy.load_state_dict(model.state_dict(), strict=False)

# Save
sb3_model.save(os.path.join(save_dir, "bc_model.zip"))
vec_norm_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
print(f"✅ SB3 model saved at: {os.path.join(save_dir, 'bc_model.zip')}")
print(f"✅ VecNormalize saved at: {os.path.join(save_dir, 'vec_normalize.pkl')}")

# Save training config
with open(os.path.join(save_dir, "training_config.pkl"), "wb") as f:
    pickle.dump({
        'training_samples': len(obs),
        'best_val_loss': best_val_loss,
        'policy_kwargs': policy_kwargs
    }, f)
print(f"✅ Training configuration saved at: {os.path.join(save_dir, 'training_config.pkl')}")
