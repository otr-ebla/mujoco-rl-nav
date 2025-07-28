
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.spaces import Box
import os, logging
from torch.utils.tensorboard import SummaryWriter
import shutil
from datetime import datetime
import argparse
import gymnasium as gym
from IL_HAMRRLN import MAX_LIN_VEL_ROBOT, NUM_RAYS, N_STACKING
from generate_IL_data import TOTAL_DATA_STEPS
from lidarCNN_extractor import LidarCNNExtractor
import matplotlib.pyplot as plt

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, input_size, hidden_size=16, num_layers=1):
        super().__init__(observation_space, features_dim=128)
        
        input_size = observation_space.shape[2]  # Flattened input size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )
        self._features_dim = 128

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)



plt.ion()  # turn on interactive plotting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============
# Utility: Safe save on interrupt
# ============
def _safe_save(policy, save_dir, suffix="interrupt"):
    """
    Save the policy weights immediately on KeyboardInterrupt.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"best_policy_{suffix}_{ts}.pt"
    path = os.path.join(save_dir, fname)
    torch.save(policy.state_dict(), path)
    logger.info(f"💾 Interrupt save: policy weights to {path}")


# ============
# Load Expert Data
# ============

MAX_LIN_VEL = MAX_LIN_VEL_ROBOT
abs_MAX_ANG_VEL = 1.0

num_rays = NUM_RAYS
n_stacking = N_STACKING
stacked_lidar_size = num_rays * n_stacking  

data_size = TOTAL_DATA_STEPS

# Load dataset
data = np.load("expert_data/expert_data.npz")
observations_raw = data["observations"]
actions_raw = data["actions"]

# Remove singleton dimension if present
if actions_raw.ndim == 3:
    actions_raw = actions_raw.squeeze(1)

print("Original actions shape:", actions_raw.shape)
print("Original observations shape:", observations_raw.shape)

# print(f"Action raw at index: {actions_raw[100]}")
# print(f"Observation raw at index: {observations_raw[100]}")

# Optional: Filter out extreme actions (if needed)
actions_index_to_filter = np.where((actions_raw[:, 0] <= MAX_LIN_VEL) & (np.abs(actions_raw[:, 1]) <= 1.0))[0]
actions = actions_raw[actions_index_to_filter]
observations = observations_raw[actions_index_to_filter]



reshaped_observations = np.empty((len(observations), n_stacking , (2 + num_rays)), dtype=np.float32)
for i in range(n_stacking):
    reshaped_observations[:, i, 0] = np.squeeze(observations[:, i:i + 1])  # distance
    reshaped_observations[:, i, 1] = np.squeeze(observations[:, i+ n_stacking:i+ n_stacking+1])  # angle
    reshaped_observations[:, i, 2:] = observations[:, i + 2*n_stacking:i  + 2*n_stacking + num_rays]  # lidar readings


observations = reshaped_observations
print("Reshaped observations shape:", observations.shape)
print("Third dimension size:", observations.shape[2])

# Action histograms
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.hist(actions[:, 0], bins=100, color='skyblue')
# plt.title("Linear Velocity Distribution")
# plt.xlabel("v (m/s)")
# plt.ylabel("Count")

# plt.subplot(1, 2, 2)
# plt.hist(actions[:, 1], bins=100, color='salmon')
# plt.title("Angular Velocity Distribution")
# plt.xlabel("ω (rad/s)")
# plt.ylabel("Count")

# plt.tight_layout()
# plt.show()


# print()
# print("Action variance:", np.var(actions_raw, axis=0))
# print()

# print()
# print()
# print("Action mean:", actions.mean(axis=0))
# print("Action std:", actions.std(axis=0))
# print("Linear vel range:", actions[:, 0].min(), actions[:, 0].max())
# print("Angular vel range:", actions[:, 1].min(), actions[:, 1].max())


# import matplotlib.pyplot as plt

# print("Filtered dataset length:", len(actions)) 

# print(f"Loaded expert data with {len(observations)} samples")
# print("Observations shape:", observations.shape)
# print("Actions shape:", actions.shape)

# polar_start = stacked_lidar_size
# polar_angle = observations[:, n_stacking:2*n_stacking]  # Extract polar angles


# angular_vel = actions[:, 1]               # shape: (N,)

# goal_angles = observations[:, 2*n_stacking-1]

# angular_vels = actions[:, 1]

# plt.figure()
# plt.scatter(goal_angles, angular_vels, s=2, alpha=0.3)
# plt.xlabel("θ (polar goal angle)")
# plt.ylabel("Angular velocity")
# plt.title("Does angular velocity match goal angle? (|ang vel| > 1e-5)")
# plt.grid(True)
# plt.show()

























# ============


class ExpertDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)
        
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

dataset = ExpertDataset(observations, actions)

# Split into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# ============
# Define Observation and Action Spaces
# ============

act_dim = actions.shape[1]
act_low = np.array([0.0, -1.0])
act_high = np.array([1.0, 1.0])
act_space = Box(low=act_low, high=act_high, dtype=np.float32)

# Observation bounds (lidar + polar goal)
# obs_low = np.concatenate([
#     np.zeros(stacked_lidar_size, dtype=np.float32),
#     np.concatenate([
#         np.zeros(n_stacking, dtype=np.float32),        # distance
#         np.full(n_stacking, -np.pi, dtype=np.float32)  # angle
#     ])
# ])
# obs_high = np.concatenate([
#     np.full(stacked_lidar_size, 200.0, dtype=np.float32),
#     np.concatenate([
#         np.full(n_stacking, 200.0, dtype=np.float32),
#         np.full(n_stacking, np.pi, dtype=np.float32)
#     ])
# ])
# Adapt observation space to match reshaped_observations shape: (n_stacking, 2 + num_rays)

obs_low = np.zeros((n_stacking, 2 + num_rays), dtype=np.float32)
obs_low[:, 1] = -np.pi  # angle dimension

obs_high = np.full((n_stacking, 2 + num_rays), 200.0, dtype=np.float32)
obs_high[:, 1] = np.pi  # angle dimension

obs_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

# ============
# Initialize Policy Network
# ============

# policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))


# policy = ActorCriticPolicy(
#     observation_space=obs_space,
#     action_space=act_space,
#     lr_schedule=lambda _: 3e-4,
#     **policy_kwargs,
# )


# policy_kwargs = {
#     'features_extractor_class': LidarCNNExtractor,
#     'features_extractor_kwargs': {
#         'n_stacking': n_stacking,
#         'num_rays': num_rays
#     },
#     'net_arch': dict(pi=[128, 128], vf=[128, 128])
# }


# policy = ActorCriticPolicy(
#     observation_space=obs_space,
#     action_space=act_space,
#     lr_schedule=lambda _: 3e-4,
#     **policy_kwargs,
# )



policy_kwargs = {
    'features_extractor_class': LSTMFeatureExtractor,
    'features_extractor_kwargs': {
        'input_size': observations.shape[1],  # lidar + polar goal
        'hidden_size': 16,
        'num_layers': 1,
    },
    'net_arch': dict(pi=[128, 128], vf=[128, 128])
}

policy = ActorCriticPolicy(
    observation_space=obs_space,
    action_space=act_space,
    lr_schedule=lambda _: 1e-4,
    **policy_kwargs,
)







# --- Hyperparameters and setup ---
optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True,
)
loss_fn = nn.MSELoss()




# Prepare save directory
save_dir = "LSTM_policy"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

print()
print(f"Loaded expert data with {len(observations)} samples")
print("Observations shape:", observations.shape)
print("Actions shape:", actions.shape)
print()
# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print("CUDA device name:", torch.cuda.get_device_name(0))
policy = policy.to(device)


# Weighted MSE setup
action_variance = torch.tensor(np.var(actions_raw, axis=0), device=device)
inv_var_weights = 1.0 / action_variance
weights = inv_var_weights / inv_var_weights.sum() * 2.0  # Normalize sum ≈ 2

# --- Real-time plotting buffers ---
train_losses = []
val_losses   = []
epochs_list  = []

# Create figure + twin axes
fig, ax1 = plt.subplots()
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")


# Empty lines
line1, = ax1.plot([], [], color="tab:blue", label="Train MSE")
line2, = ax1.plot([], [], color="tab:red",   label="Val MSE")




# Legends
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

plt.tight_layout()
# Show non-blocking
plt.ion()
fig.show()
fig.canvas.draw()

patience = 8
best_val_loss   = float("inf")
no_improve      = 0
improve_counter = 0


policy.train()  # Set policy to training mode
try: 
    for epoch in range(1, 1001):
        # --- Training step ---
        policy.train()
        train_mse_accum = 0.0

        for batch_obs, batch_act in train_loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            features = policy.extract_features(batch_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            dist = policy._get_action_dist_from_latent(latent_pi)

            # Use distribution mode (mean) for prediction
            pred = dist.distribution.mean
            loss = loss_fn(pred, batch_act)
            


    
            # goal_angle = batch_obs[:, stacked_lidar_size + 1]  # shape (B,)
            # target_angvel = goal_angle / np.pi  # same range as action space
            # target_angvel = torch.clamp(target_angvel, -1.0, 1.0)
            # angle_loss = nn.functional.mse_loss(pred[:, 1], target_angvel)
            # bc_loss = loss_fn(pred, batch_act)
            # loss = bc_loss + 3 * angle_loss


            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            train_mse_accum += loss.item()

        avg_train = train_mse_accum / len(train_loader)

        # --- Validation step ---
        policy.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            policy.eval()
            for batch_obs, batch_act in val_loader:
                batch_obs = batch_obs.to(device)
                batch_act = batch_act.to(device)

                features = policy.extract_features(batch_obs)
                latent_pi, _ = policy.mlp_extractor(features)
                dist = policy._get_action_dist_from_latent(latent_pi)
                pred = dist.distribution.mean

                val_mse_accum += loss_fn(pred, batch_act).item()
                # mse = (pred - batch_act) ** 2
                # val_loss = torch.mean(mse*weights).item()  # Weighted MSE
                # val_mse_accum += val_loss

        avg_val = val_mse_accum / len(val_loader)
        policy.train()  # Set policy back to training mode
        # --- Logging ---
        print(f"Epoch {epoch:04d}: Train MSE = {avg_train:.6f}, Val MSE = {avg_val:.6f}")

        # --- Scheduler ---
        scheduler.step(avg_val)

        # --- Early stopping & checkpointing ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            improve_counter += 1
            if improve_counter % 10 == 0:
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'n_stacking': n_stacking,
                    'num_rays': num_rays,
                    'policy_kwargs': policy_kwargs,
                    'observation_space': obs_space,
                    #'obs_mean': obs_mean,
                    #'obs_std': obs_std,
                    'action_space': act_space
                }, os.path.join(save_dir, "best_policy.pt"))
                print(f"✅ Checkpoint saved (val loss {best_val_loss:.6f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

        # --- Update real-time plot ---
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        epochs_list.append(epoch)

        line1.set_data(epochs_list, train_losses)
        line2.set_data(epochs_list, val_losses)
        ax1.relim(); ax1.autoscale_view()
        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(0.01)

except KeyboardInterrupt:
    logger.warning("⏹ CTRL+C detected—saving interim model...")
    _safe_save(policy, save_dir)


print("✅ Training complete")
plt.ioff()
plt.show()
print(f"Best validation loss: {best_val_loss:.6f}")


# ============
# Create SB3 Compatible Model (Only Once After Training)
# ============

print("Creating SB3 compatible model...")

# Create dummy environment matching observation and action space
class DummyEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

# ✅ Load best policy weights
checkpoint = torch.load(os.path.join(save_dir, "best_policy.pt"))

# Wrap dummy env with VecNormalize
dummy_env = DummyEnv(obs_space, act_space)
vec_norm_env = VecNormalize(DummyVecEnv([lambda: dummy_env]), norm_obs=True, norm_reward=False)
vec_norm_env.training = False

# Reconstruct SB3 model (standard MLP architecture)
sb3_model = PPO(
    policy=LSTMFeatureExtractor,
    env=vec_norm_env,
    policy_kwargs=policy_kwargs,
    device="cpu"
)
sb3_model.policy.load_state_dict(checkpoint["policy_state_dict"])

# Save final models
sb3_model.save(os.path.join(save_dir, "bc_model.zip"))
vec_norm_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
print(f"✅ SB3 model saved at: {os.path.join(save_dir, 'bc_model.zip')}")
print(f"✅ VecNormalize saved at: {os.path.join(save_dir, 'vec_normalize.pkl')}")

# Save training config
import pickle
with open(os.path.join(save_dir, "training_config.pkl"), "wb") as f:
    pickle.dump({
        'training_samples': len(observations),
        'best_val_loss': best_val_loss,
    }, f)
print(f"✅ Training configuration saved at: {os.path.join(save_dir, 'training_config.pkl')}")

