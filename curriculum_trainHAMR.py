import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv,VecNormalize
import argparse
import xml.etree.ElementTree as ET
import os
import torch
from assets.custom_callback import RewardCallback
from HAMRRLN import hamrrln
#from nohumans_HAMRRLN import nohumans_hamrrln
from stable_baselines3.common.callbacks import BaseCallback
from assets.custompolicy import TanhActorCriticPolicy
from torch.utils.tensorboard import SummaryWriter

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from trainHAMR import hamrrln, ScenarioSuccessCallback
from assets.custompolicy import TanhActorCriticPolicy

def make_env(num_rays, model_path="assets/world.xml", training=True, n_humans=5, n_stacking=20):
    def _init():
        env = hamrrln(
            num_rays=num_rays,
            model_path=model_path,
            training=training,
            n_humans=n_humans,
            n_stacking=n_stacking
        )
        return env
    return _init

def get_model_class(trainer_name):
    if trainer_name == "PPO":
        return PPO
    elif trainer_name == "SAC":
        return SAC
    elif trainer_name == "TD3":
        return TD3
    elif trainer_name == "TQC":
        return TQC
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}")

def continue_training(args):
    log_dir = "./TENSORBOARD/"
    os.makedirs(log_dir, exist_ok=True)

    # Setup environment
    env = SubprocVecEnv([make_env(args.num_rays, args.model_path, training=True, n_humans=args.n_humans) for _ in range(args.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Load model
    ModelClass = get_model_class(args.trainer)
    model = ModelClass.load(args.model_path_zip, env=env)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path="./policy_checkpoints/",
        name_prefix=args.run_id
    )
    reward_callback = RewardCallback()
    scenario_success_callback = ScenarioSuccessCallback(log_freq=50000)

    callbacks = [checkpoint_callback, reward_callback, scenario_success_callback]

    # Resume training
    print(f"Resuming training with {args.trainer} for {args.num_steps} steps...")
    model.learn(
        total_timesteps=args.num_steps,
        callback=callbacks,
        tb_log_name=args.run_id
    )

    # Save final model and VecNormalize
    model.save(f"{args.run_id}_continued")
    env.save(os.path.join(log_dir, f"{args.run_id}_continued.pkl"))
    print("\n=== FINAL TRAINING SCENARIO STATISTICS ===")
    scenario_success_callback._log_scenario_summary()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training from a saved model for curriculum learning.")
    parser.add_argument("--model_path_zip", type=str, required=True, help="Path to the .zip file of the saved model.")
    parser.add_argument("--trainer", type=str, default="TQC", help="Trainer to use (PPO, SAC, TD3, TQC).")
    parser.add_argument("--num_steps", type=int, default=1000000, help="Number of steps to continue training.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--num_rays", type=int, default=108, help="Number of LiDAR rays.")
    parser.add_argument("--n_humans", type=int, default=5, help="Number of human agents.")
    parser.add_argument("--model_path", type=str, default="assets/world.xml", help="MuJoCo model path.")
    parser.add_argument("--run_id", type=str, default="curriculum_run", help="Run ID for TensorBoard logging.")
    args = parser.parse_args()

    continue_training(args)
