#!/usr/bin/env python3
"""
SAC training script for the LidarNav2DEnv (MuJoCo + 2D LiDAR).
- Vectorized envs + observation normalization
- Evaluation + checkpointing callbacks
- Tuned defaults for continuous actions [v, ω]

Usage (from project root):
    python train_lidar_nav_sac.py \
        --total-steps 3_000_000 \
        --n-envs 8 \
        --logdir runs/sac_lidar \
        --model-name sac_lidar_nav

Requires:
    pip install stable-baselines3[extra] gymnasium mujoco torch tensorboard

Note:
    Ensure the module `lightHAMRRLN_lidar2d_adapted.py` (with class LidarNav2DEnv)
    is on PYTHONPATH or in the same folder.
"""

import os
import argparse
from typing import Callable

import numpy as np
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise

# Import your env
from src.core.lidar_adaptedHAMMRLN import LidarNav2DEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--total-steps', type=int, default=2_000_000)
    p.add_argument('--n-envs', type=int, default=4)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--eval-freq', type=int, default=50_000)
    p.add_argument('--checkpoint-freq', type=int, default=250_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--logdir', type=str, default='runs/sac_lidar')
    p.add_argument('--model-name', type=str, default='sac_lidar_nav')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=1_000_000)
    p.add_argument('--train-freq', type=int, default=1)
    p.add_argument('--gradient-steps', type=int, default=1)
    p.add_argument('--target-entropy', type=float, default=-2.0)  # ~ -|A|
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--ent-coef', type=str, default='auto')
    p.add_argument('--clip-reward', action='store_true')
    p.add_argument('--obs-norm', action='store_true', help='Enable VecNormalize for observations')
    p.add_argument('--rewards-norm', action='store_true', help='Enable VecNormalize for rewards')
    p.add_argument('--no-subproc', action='store_true', help='Use DummyVecEnv instead of SubprocVecEnv')
    # NEW: live rendering
    p.add_argument('--render-training', action='store_true', help='Show MuJoCo viewer during training (forces DummyVecEnv)')
    p.add_argument('--render-freq', type=int, default=1, help='Render every N environment steps (with --render-training)')
    return p.parse_args()


def make_env_fn(training: bool = True, render_mode: str | None = None) -> Callable:
    def _thunk():
        env = LidarNav2DEnv(training=training, render_mode=render_mode, directional_lock=True, safety_shield=True)
        return env
    return _thunk


def build_vec_env(n_envs: int, training: bool, seed: int, use_subproc: bool, render_mode: str | None):
    set_random_seed(seed)
    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    env = make_vec_env(make_env_fn(training, render_mode), n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)
    return env


def maybe_wrap_norm(env, obs_norm: bool, rewards_norm: bool, training: bool, gamma: float):
    if not (obs_norm or rewards_norm):
        return env
    env = VecNormalize(env, training=training, norm_obs=obs_norm, norm_reward=rewards_norm, gamma=gamma, clip_obs=5.0)
    return env
    env = VecNormalize(env, training=training, norm_obs=obs_norm, norm_reward=rewards_norm, gamma=gamma, clip_obs=5.0)
    return env


def main():
    args = parse_args()

    # Enforce single env if rendering is enabled
    if args.render_training:
        if args.n_envs != 1:
            print("[INFO] --render-training attivo: imposto --n-envs=1 e --no-subproc")
        args.n_envs = 1
        args.no_subproc = True

    os.makedirs(args.logdir, exist_ok=True)

    # Select vectorization mode & render
    use_subproc = (not args.no_subproc) and (not args.render_training)
    render_mode = 'human' if args.render_training else None

    # === Train env ===
    train_env = build_vec_env(args.n_envs, training=True, seed=args.seed, use_subproc=use_subproc, render_mode=render_mode)
    train_env = maybe_wrap_norm(train_env, obs_norm=args.obs_norm, rewards_norm=args.rewards_norm, training=True, gamma=args.gamma)

    # === Eval env ===
    eval_env = build_vec_env(1, training=False, seed=args.seed + 42, use_subproc=False, render_mode=None)
    eval_env = maybe_wrap_norm(eval_env, obs_norm=args.obs_norm, rewards_norm=False, training=False, gamma=args.gamma)

    # If we use VecNormalize, sync stats between train and eval
    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms

    # Logger
    new_logger = configure(folder=args.logdir, format_strings=["stdout", "tensorboard", "csv"])

    # Action noise (optional)
    action_noise = None

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=th.nn.SiLU,
        normalize_images=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,          # 'auto' or float
        target_entropy=args.target_entropy,
        action_noise=action_noise,
        verbose=1,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )
    model.set_logger(new_logger)

    # === Live Render Callback (optional) ===
    class LiveRenderCallback(BaseCallback):
        def __init__(self, render_every: int = 1):
            super().__init__()
            self.render_every = max(1, int(render_every))
        def _on_step(self) -> bool:
            if not args.render_training:
                return True
            if (self.n_calls % self.render_every) != 0:
                return True
            try:
                # unwrap to the first underlying env
                env = self.training_env
                # VecNormalize -> VecEnv -> [envs[0]] -> underlying env
                if hasattr(env, 'venv'):
                    env = env.venv
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    base = env.envs[0]
                    # Some wrappers have .env attribute
                    while hasattr(base, 'env'):
                        base = base.env
                    if hasattr(base, 'render'):
                        base.render('human')
            except Exception:
                pass
            return True

    # === Callbacks ===
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, 'best'),
        log_path=os.path.join(args.logdir, 'eval'),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(args.logdir, 'ckpts'),
        name_prefix=args.model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # === Train ===
    cbs = [eval_callback, checkpoint_callback]
    if args.render_training:
        cbs.append(LiveRenderCallback(args.render_freq))
    model.learn(total_timesteps=args.total_steps, log_interval=10, callback=cbs)

    # Save final
    final_path = os.path.join(args.logdir, f"{args.model_name}_final")
    model.save(final_path)

    # Save VecNormalize stats (if any)
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(args.logdir, 'vecnormalize_train.pkl'))
        # copy obs_rms to eval env
        if isinstance(eval_env, VecNormalize):
            eval_env.obs_rms = train_env.obs_rms

    # === Quick test run ===
    print("\nEvaluating final policy...")
    ep_rets = []
    for _ in range(args.eval_episodes):
        obs, _ = eval_env.reset()
        done, trunc = False, False
        ep_ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = eval_env.step(action)
            ep_ret += float(reward)
        ep_rets.append(ep_ret)
    print(f"Eval returns: mean={np.mean(ep_rets):.2f} ± {np.std(ep_rets):.2f} over {len(ep_rets)} eps")

    # Close
    train_env.close(); eval_env.close()


if __name__ == '__main__':
    main()
