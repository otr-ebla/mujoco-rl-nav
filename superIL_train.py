#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Imitation Learning (Behavior Cloning) trainer for HAMRRLN.

Goals
-----
- Train a policy purely with IL (supervised learning on (obs, action)).
- Save a clean raw `state_dict` at `bc_policy/best_policy.pt` (safe with PyTorch 2.6+).
- Also export an SB3 `PPO` model `bc_policy/bc_model.zip` and matching
  `bc_policy/vec_normalize.pkl` so it can be:
  • Evaluated from `trainHAMR.py` with `--trainer BC`, or
  • Used as initialization for RL fine-tuning (PPO/SAC/TD3/TQC).

Key features
------------
- Deterministic seeding
- CLI for data path, output dir, hyperparams, workers, AMP, etc.
- Early stopping + ReduceLROnPlateau
- Optional auxiliary angle-consistency loss (align ω with goal angle)
- No pickling of Gym spaces in checkpoints (avoids `weights_only=True` errors)
- Headless-safe plotting (disabled by default)

Dataset assumptions
-------------------
`expert_data.npz` contains:
- observations: (N, obs_dim) float32
- actions: (N, 2) float32 where [lin in [0,1], ang in [-1,1]]



HOW TO TRAIN: python3 IL_train.py --data expert_data/expert_data.npz --out bc_policy \
  --batch 512 --epochs 500 --patience 6 --lr 2e-3 --wd 1e-5 --workers 6 --amp



"""
from __future__ import annotations
import os, json, argparse, shutil, warnings
from dataclasses import dataclass
from typing import Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import gymnasium as gym
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Project-specific constants
from IL_HAMRRLN import MAX_LIN_VEL_ROBOT, NUM_RAYS, N_STACKING


# ------------------------
# Utilities
# ------------------------

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class TrainStats:
    best_val: float = float("inf")
    no_improve: int = 0
    epochs_done: int = 0


class ExpertDataset(Dataset):
    def __init__(self, obs: np.ndarray, acts: np.ndarray):
        self.obs = torch.from_numpy(obs.astype(np.float32))
        self.acts = torch.from_numpy(acts.astype(np.float32))

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int):
        return self.obs[idx], self.acts[idx]


# ------------------------
# Core training
# ------------------------

def build_spaces(num_rays: int, n_stacking: int) -> Tuple[Box, Box]:
    stacked_lidar_size = num_rays * n_stacking
    obs_low = np.concatenate([
        np.zeros(n_stacking, dtype=np.float32),             # goal dist
        np.full(n_stacking, -np.pi, dtype=np.float32),      # goal angle
        np.zeros(stacked_lidar_size, dtype=np.float32),     # lidar
    ])
    obs_high = np.concatenate([
        np.full(n_stacking, 200.0, dtype=np.float32),
        np.full(n_stacking, np.pi, dtype=np.float32),
        np.full(stacked_lidar_size, 200.0, dtype=np.float32),
    ])
    obs_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

    act_low = np.array([0.0, -1.0], dtype=np.float32)
    act_high = np.array([1.0, 1.0], dtype=np.float32)
    act_space = Box(low=act_low, high=act_high, dtype=np.float32)
    return obs_space, act_space


def angle_aux_loss(obs_batch: torch.Tensor, pred_actions: torch.Tensor, num_rays: int, n_stacking: int) -> torch.Tensor:
    """Encourage ω to align with latest goal angle (scaled to [-1,1])."""
    stacked_lidar_size = num_rays * n_stacking
    # angles occupy the second block of size n_stacking; take the most recent one
    angle_idx = stacked_lidar_size + (n_stacking - 1)
    goal_angle = obs_batch[:, angle_idx].clamp(-np.pi, np.pi)
    target_ang = (goal_angle / np.pi).clamp(-1.0, 1.0)
    return nn.functional.mse_loss(pred_actions[:, 1], target_ang)


def train_bc(
    data_path: str,
    save_dir: str = "bc_policy",
    batch_size: int = 512,
    epochs: int = 500,
    patience: int = 10,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    net_pi: Tuple[int, ...] = (128, 128),
    net_vf: Tuple[int, ...] = (128, 128),
    aux_angle_lambda: float = 0.0,
    amp: bool = False,
    workers: int = 4,
    seed: int = 42,
    plot: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Seeding & device
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC] Device: {device}")

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path)
    observations = data["observations"]
    actions = data["actions"]
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions[:, 0, :]

    # Filter out-of-range actions
    keep = (actions[:, 0] <= MAX_LIN_VEL_ROBOT + 1e-6) & (np.abs(actions[:, 1]) <= 1.0 + 1e-6)
    observations = observations[keep]
    actions = actions[keep]

    print(f"[BC] Data: obs={observations.shape} acts={actions.shape}")
    print(f"[BC] Action mean={actions.mean(0)} std={actions.std(0)}")

    # Build spaces
    obs_space, act_space = build_spaces(NUM_RAYS, N_STACKING)

    # Build SB3 ActorCriticPolicy to reuse its feature extractor and action head
    policy_kwargs = dict(net_arch=dict(pi=list(net_pi), vf=list(net_vf)))
    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-4,   # not used in BC; optimizer below drives LR
        **policy_kwargs,
    ).to(device)

    # Dataset & loaders
    ds = ExpertDataset(observations, actions)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    # Optim & scheduler
    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    loss_fn = nn.MSELoss()

    # Per-dimension weighting by inverse variance for stability
    action_var = np.var(actions, axis=0) + 1e-8
    inv_var = 1.0 / action_var
    dim_weights = torch.tensor(inv_var / inv_var.sum(), dtype=torch.float32, device=device)  # sums to 1
    print(f"[BC] Dim weights (lin, ang): {dim_weights.tolist()}")

    # Training loop
    stats = TrainStats()
    best_path = os.path.join(save_dir, "best_policy.pt")

    if plot:
        try:
            import matplotlib
            if os.environ.get("HEADLESS", "1") == "1":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots()
            tr_hist, va_hist = [], []
        except Exception as e:
            warnings.warn(f"Plot disabled: {e}")
            plot = False

    try:
        for epoch in range(1, epochs + 1):
            policy.train()
            train_loss_sum = 0.0
            for obs_b, act_b in train_loader:
                obs_b = obs_b.to(device, non_blocking=pin)
                act_b = act_b.to(device, non_blocking=pin)

                with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                    features = policy.extract_features(obs_b)
                    latent_pi, _ = policy.mlp_extractor(features)
                    dist = policy._get_action_dist_from_latent(latent_pi)
                    pred = dist.distribution.mean

                    base_mse = loss_fn(pred, act_b)
                    per_dim = ((pred - act_b) ** 2).mean(0)
                    weighted = (per_dim * dim_weights).sum()
                    loss = 0.5 * base_mse + 0.5 * weighted
                    if aux_angle_lambda > 0.0:
                        loss = loss + aux_angle_lambda * angle_aux_loss(obs_b, pred, NUM_RAYS, N_STACKING)

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()

                train_loss_sum += float(loss.detach())

            avg_train = train_loss_sum / max(1, len(train_loader))

            # Validation
            policy.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for obs_b, act_b in val_loader:
                    obs_b = obs_b.to(device, non_blocking=pin)
                    act_b = act_b.to(device, non_blocking=pin)
                    features = policy.extract_features(obs_b)
                    latent_pi, _ = policy.mlp_extractor(features)
                    dist = policy._get_action_dist_from_latent(latent_pi)
                    pred = dist.distribution.mean

                    base_mse = loss_fn(pred, act_b)
                    per_dim = ((pred - act_b) ** 2).mean(0)
                    weighted = (per_dim * dim_weights).sum()
                    vloss = 0.5 * base_mse + 0.5 * weighted
                    if aux_angle_lambda > 0.0:
                        vloss = vloss + aux_angle_lambda * angle_aux_loss(obs_b, pred, NUM_RAYS, N_STACKING)

                    val_loss_sum += float(vloss)

            avg_val = val_loss_sum / max(1, len(val_loader))
            scheduler.step(avg_val)

            stats.epochs_done = epoch
            print(f"[BC][{epoch:04d}] train={avg_train:.6f}  val={avg_val:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping + checkpoint
            if avg_val + 1e-6 < stats.best_val:
                stats.best_val = avg_val
                stats.no_improve = 0
                torch.save(policy.state_dict(), best_path)  # RAW state_dict only
            else:
                stats.no_improve += 1
                if stats.no_improve >= patience:
                    print(f"[BC] Early stop at epoch {epoch}")
                    break

            # Optional plot
            if plot:
                tr_hist.append(avg_train); va_hist.append(avg_val)
                ax.clear()
                ax.plot(tr_hist, label="train")
                ax.plot(va_hist, label="val")
                ax.set_title("BC Training")
                ax.set_xlabel("epoch")
                ax.set_ylabel("loss")
                ax.legend(loc="upper right")
                fig.canvas.draw(); fig.canvas.flush_events()

    except KeyboardInterrupt:
        tmp = os.path.join(save_dir, f"best_policy_interrupt_{nowstamp()}.pt")
        torch.save(policy.state_dict(), tmp)
        print(f"[BC] ⚠️ Interrupted. Interim weights saved to {tmp}")

    print(f"[BC] Best validation loss: {stats.best_val:.6f}")

    # ------------------------
    # Export SB3 PPO model for evaluation & RL fine-tuning
    # ------------------------
    class DummyEnv(gym.Env):
        def __init__(self, obs_space_: Box, act_space_: Box):
            super().__init__()
            self.observation_space = obs_space_
            self.action_space = act_space_
        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}
        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    dummy_env = DummyVecEnv([lambda: DummyEnv(obs_space, act_space)])
    vec_norm = VecNormalize(dummy_env, norm_obs=False, norm_reward=False)
    vec_norm.training = False

    ppo = PPO(policy=ActorCriticPolicy, env=vec_norm, policy_kwargs=policy_kwargs, device="cpu", verbose=0)

    # Load raw state_dict
    state_dict = torch.load(best_path, map_location="cpu")
    missing, unexpected = ppo.policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[BC] State dict load: missing={missing} unexpected={unexpected}")

    # Save
    ppo.save(os.path.join(save_dir, "bc_model.zip"))
    vec_norm.save(os.path.join(save_dir, "vec_normalize.pkl"))

    # Minimal metadata (JSON)
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump({
            "data_path": data_path,
            "best_val_loss": float(stats.best_val),
            "samples": int(observations.shape[0]),
            "policy_kwargs": policy_kwargs,
            "n_stacking": int(N_STACKING),
            "num_rays": int(NUM_RAYS),
            "timestamp": nowstamp(),
        }, f, indent=2)

    print("[BC] ✅ Exported: bc_policy/bc_model.zip, vec_normalize.pkl, best_policy.pt")


# ------------------------
# CLI
# ------------------------

def main():
    p = argparse.ArgumentParser(description="Behavior Cloning trainer (SB3-compatible export)")
    p.add_argument("--data", type=str, default="expert_data/expert_data.npz", help="Path to expert data .npz")
    p.add_argument("--out", type=str, default="bc_policy", help="Output directory")
    p.add_argument("--batch", type=int, default=512, help="Batch size")
    p.add_argument("--epochs", type=int, default=500, help="Max epochs")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    p.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--net_pi", type=int, nargs="+", default=[128, 128], help="Actor MLP sizes")
    p.add_argument("--net_vf", type=int, nargs="+", default=[128, 128], help="Critic MLP sizes")
    p.add_argument("--aux_angle", type=float, default=0.0, help="Aux angle-consistency loss weight (0=off)")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--plot", action="store_true", help="Live plot (may not work headless)")
    args = p.parse_args()

    # Fresh output
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    train_bc(
        data_path=args.data,
        save_dir=args.out,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.wd,
        net_pi=tuple(args.net_pi),
        net_vf=tuple(args.net_vf),
        aux_angle_lambda=args.aux_angle,
        amp=args.amp,
        workers=args.workers,
        seed=args.seed,
        plot=args.plot,
    )


if __name__ == "__main__":
    # Silence jax warning if present
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
