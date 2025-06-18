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
from stable_baselines3.common.callbacks import BaseCallback
from assets.custompolicy import TanhActorCriticPolicy
 


import os
os.environ['JAX_PLATFORMS'] = 'cpu'

class ScenarioSuccessCallback(BaseCallback):
    """
    Custom callback to track success rates for each scenario during training.
    """
    def __init__(self, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.scenario_stats = {}  # Will store stats for each scenario
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if any environment has finished an episode
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        if len(dones) > 0 and len(infos) > 0:
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and info:
                    self._log_episode_result(info)
        
        # Log summary statistics periodically
        if self.n_calls % self.log_freq == 0:
            self._log_scenario_summary()
            
        return True
    
    def _log_episode_result(self, info):
        """Log the result of a completed episode"""
        scenario_id = info.get('scenario_id', 'unknown')
        episode_result = info.get('episode_result', 'unknown')
        
        # Initialize scenario stats if not seen before
        if scenario_id not in self.scenario_stats:
            self.scenario_stats[scenario_id] = {
                'total_episodes': 0,
                'successes': 0,
                'collisions': 0,
                'timeouts': 0
            }
        
        # Update stats
        stats = self.scenario_stats[scenario_id]
        stats['total_episodes'] += 1
        
        if episode_result == 'success':
            stats['successes'] += 1
        elif episode_result == 'collision':
            stats['collisions'] += 1
        elif episode_result == 'timeout':
            stats['timeouts'] += 1
            
        self.episode_count += 1
    
    def _log_scenario_summary(self):
        """Log summary statistics for all scenarios"""
        if not self.scenario_stats:
            return
            
        print(f"\n=== Scenario Success Rates (Step {self.num_timesteps}) ===")
        
        total_episodes_all = 0
        total_successes_all = 0
        
        for scenario_id, stats in sorted(self.scenario_stats.items()):
            if stats['total_episodes'] > 0:
                success_rate = stats['successes'] / stats['total_episodes']
                collision_rate = stats['collisions'] / stats['total_episodes']
                timeout_rate = stats['timeouts'] / stats['total_episodes']
                
                print(f"Scenario {scenario_id}: "
                      f"Episodes={stats['total_episodes']}, "
                      f"Success={success_rate:.3f}, "
                      f"Collision={collision_rate:.3f}, "
                      f"Timeout={timeout_rate:.3f}")
                
                # Log to tensorboard if available
                if hasattr(self, 'logger') and self.logger:
                    self.logger.record(f"scenario_{scenario_id}/success_rate", success_rate)
                    self.logger.record(f"scenario_{scenario_id}/collision_rate", collision_rate)
                    self.logger.record(f"scenario_{scenario_id}/timeout_rate", timeout_rate)
                    self.logger.record(f"scenario_{scenario_id}/total_episodes", stats['total_episodes'])
                
                total_episodes_all += stats['total_episodes']
                total_successes_all += stats['successes']
        
        # Overall success rate
        if total_episodes_all > 0:
            overall_success_rate = total_successes_all / total_episodes_all
            print(f"Overall Success Rate: {overall_success_rate:.3f} ({total_successes_all}/{total_episodes_all})")
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.record("training/overall_success_rate", overall_success_rate)
        
        print("=" * 50)

class PolicySaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"policy_step_{self.num_timesteps}.zip")
            self.model.save(save_file)
            if self.verbose:
                print(f"✅ Saved model at step {self.num_timesteps} to {save_file}")
        return True



def make_env(num_rays, model_path="assets/world.xml", training = True, n_humans = 5, render_mode=None, n_stacking = 20):
    def _init():
        env = hamrrln(
            num_rays=num_rays, 
            model_path=model_path, 
            training=training,
            n_humans = n_humans,
            n_stacking=n_stacking,
            )
        return env
    return _init



def train_agent(num_rays, model_path="assets/world.xml", num_envs=16, num_steps=100000, run_id="training1", training=True, trainer="ppo", n_humans=5, n_stacking=10):
    log_dir = "./TENSORBOARD/"
    os.makedirs(log_dir, exist_ok=True)

    if not training:
        env = hamrrln(
            num_rays=num_rays, 
            model_path=model_path, 
            training=False,
            n_humans = n_humans,
            n_stacking=n_stacking,
            )
    else:
        # Create vectorized environment
        env = SubprocVecEnv([make_env(num_rays, model_path, training=training) for _ in range(num_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

 
        
    policy_kwargs = dict(
        net_arch=[128, 128],
        log_std_init=-2.0,
    )

    # Create PPO model with built-in logging
    if trainer == "PPO":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )
    elif trainer == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cpu",
            learning_rate=0.0001,    # Kept same
            buffer_size=1000000,     # Kept same
            learning_starts=5000,    # Increased from 1000
            batch_size=512,          # Increased from 256
            tau=0.01,               # Increased from 0.005
            gamma=0.99,             # Kept same
            train_freq=1,           # Kept same
            gradient_steps=1,       # Kept same
            ent_coef="auto",        # Kept same
            policy_kwargs=policy_kwargs
        )
    elif trainer == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cpu",
            learning_rate=0.0003,  # Reduced from 0.001
            buffer_size=1000000,   # Kept same
            learning_starts=20000,  
            batch_size=512,        # Reduced from 256
            tau=0.005,            # Kept same
            gamma=0.98,           # Slightly reduced from 0.99
            #train_freq=(num_envs*100, "step"),    # Explicit step/epoch setting
            policy_kwargs=dict(
                net_arch=[256, 256],
                #noise_std=0.2,
                #noise_clip=0.5
            )
        )
    elif trainer == "TQC":
        model = TQC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cpu",
            learning_rate=0.0001,       # Balanced learning rate
            batch_size=512,            # Larger batch for stability
            gamma=0.98,                # Good for mid-horizon tasks
            tau=0.005,                 # Default target update rate
            ent_coef="auto_0.01",    # Adaptive entropy coefficient
            #n_quantiles=25,            # Default quantile count
            #top_quantiles_to_drop=2,   # Default truncation
            policy_kwargs=policy_kwargs
        )

     
    
    # Only keep necessary callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=2000000,
        save_path="./policy_checkpoints/",
        name_prefix=run_id
    )

    reward_callback = RewardCallback()
    scenario_success_callback = ScenarioSuccessCallback(log_freq=50000, verbose=1)



    callbacks = [checkpoint_callback, reward_callback, scenario_success_callback]
    
    # Train the model
    print(f"Training PPO agent for {num_steps} steps...")
    model.learn(
        total_timesteps=num_steps,
        callback=callbacks,
        tb_log_name=run_id  # This ensures logs go to the correct subdirectory
    )

    
    # Save the final model
    model.save(f"{run_id}")
    env.save(os.path.join(log_dir, f"{run_id}.pkl"))

    print("\n=== FINAL TRAINING SCENARIO STATISTICS ===")
    scenario_success_callback._log_scenario_summary()

    env.close()










    
if __name__ == "__main__":
    train = True

    name = "DEFAULT"

    parser = argparse.ArgumentParser(description="Train a PPO agent to navigate a cube to a target sphere.")
    parser.add_argument("--num_rays", type=int, default=108, help="Number of LiDAR rays around the sphere.")
    parser.add_argument("--model_path", type=str, default="assets/world.xml", help="Path to the MuJoCo model XML file.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
    parser.add_argument("--train", action="store_true", help="Train the PPO agent.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the PPO agent.")
    parser.add_argument("--num_steps", type=int, default=10000000, help="Number of training steps.")
    parser.add_argument("--run_id", type=str, default="DEFAULT", help="Run ID for TensorBoard logging.")
    parser.add_argument("--trainer", type=str, default="TQC", help="Trainer to use (PPO, SAC, TD3, TQC).")
    parser.add_argument("--num_obst", type=int, default=51, help="Number of obstacles in the environment.")
    args = parser.parse_args()
    
    name = args.run_id

    train = args.train
    if args.eval:
        train = False

    if train:
        train_agent(args.num_rays, args.model_path, args.num_envs, args.num_steps, args.run_id, training=True, trainer=args.trainer)
    
    # Load the last trained model with the correct algorithm class
    if args.trainer == "PPO":
        model = PPO.load(f"{name}")
    elif args.trainer == "SAC":
        model = SAC.load(f"{name}")
    elif args.trainer == "TD3":
        model = TD3.load(f"{name}")
    elif args.trainer == "TQC":
        model = TQC.load(f"{name}")
    
    eval_env = DummyVecEnv([lambda: hamrrln(
                                    num_rays=args.num_rays, 
                                    model_path=args.model_path, 
                                    training=False,
                                    n_humans = 5,
                                    n_stacking= 20,
                            )])

    # Carica la normalizzazione su DummyVecEnv
    if os.path.exists(os.path.join("./TENSORBOARD/", f"{args.run_id}.pkl")):
        eval_env = VecNormalize.load(os.path.join("./TENSORBOARD/", f"{args.run_id}.pkl"), eval_env)
    else:
        eval_env = VecNormalize.load(os.path.join("./TENSORBOARD/", "vecnormalize.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    # Evaluate the trained agent
    mean_reward, std_reward, linear_velocities, angular_velocities = evaluate_policy(model, eval_env, n_eval_episodes=20000000, deterministic=True, print_actions_means=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    eval_env.close()

    # import matplotlib.pyplot as plt
    
    # linear_velocities = np.array(linear_velocities)
    # angular_velocities = np.array(angular_velocities)

    # print(len(linear_velocities))
    # linear_velocities = 0.31*linear_velocities
    # angular_velocities = 1.9*angular_velocities
    
    # plt.figure(figsize=(8, 6))
    # plt.plot([0.2, 0.31], [1.9, 0.0], 'ro-', linewidth=2)
    # plt.scatter(linear_velocities, angular_velocities, alpha=0.2)
    # plt.xlabel("Linear Velocity")
    # plt.ylabel("Angular Velocity")
    # plt.title(f"2D Scatter Plot of Linear vs Angular Velocities with {len(linear_velocities)} points")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("velocity_scatter.png")
    # plt.show()