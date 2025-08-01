import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3, A2C
from sb3_contrib import TQC

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv,VecNormalize 
import argparse
import xml.etree.ElementTree as ET
import os
import torch, pickle
from assets.custom_callback import RewardCallback
from HAMRRLN import hamrrln, N_HUMANS
from stable_baselines3.common.callbacks import BaseCallback
from assets.custompolicy import TanhActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

from IL_HAMRRLN import NUM_RAYS, N_STACKING
import logging
from assets.train_classes import ScenarioSuccessCallback, PolicySaveCallback, RenderCallback

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# ============
# Logging Setup
# ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 


def _safe_save(model, env, run_id, log_dir, suffix="interrupt"):
    """
    Save the model and VecNormalize statistics with a suffix so we
    don’t overwrite the final checkpoint.
    """
    timesteps = getattr(model, "num_timesteps", "0")
    stamp     = f"{suffix}_{timesteps}"
    model_path = f"{run_id}_{stamp}.zip"
    norm_path  = os.path.join(log_dir, f"{run_id}_{stamp}.pkl")

    model.save(model_path)
    if isinstance(env, VecNormalize):
        env.save(norm_path)

    print(f"💾 Saved model to {model_path}")
    if isinstance(env, VecNormalize):
        print(f"💾 Saved VecNormalize to {norm_path}")
 




def make_env(num_rays, model_path="assets/world.xml", training = True, n_humans = 5, render_mode=None, n_stacking = 10, stacking=True):
    def _init():
        env = hamrrln(
            num_rays=num_rays, 
            model_path=model_path, 
            training=training,
            n_humans = n_humans,
            n_stacking=n_stacking,
            enable_stacking=stacking,
            )
        return env
    return _init



def train_agent(num_rays, 
                model_path="assets/world.xml", 
                num_envs=16, 

                num_steps=100000, 
                run_id="training1", 
                training=True, 

                trainer="ppo",  
                stacking=True,
                n_stacking=10,
                cl_resume = False,
                bc_policy_path="bc_policy/",
                render_training=False):
    
    n_humans = N_HUMANS
    
    log_dir = "./TENSORBOARD/"
    os.makedirs(log_dir, exist_ok=True)

    if not training:
        env = hamrrln(
            num_rays=num_rays, 
            model_path=model_path, 
            training=False,
            n_humans = n_humans,
            n_stacking=n_stacking,
            enable_stacking=stacking,
            )
    else:
        # Create vectorized environment
        if render_training and num_envs == 1:
            env = DummyVecEnv([ lambda: hamrrln(
                num_rays=num_rays,
                model_path=model_path,
                training=training,
                n_humans=n_humans,
                n_stacking=n_stacking,
                enable_stacking=stacking,
                render_mode="human",  # For rendering during training
                )])



        # STANDARD RL TRAINING ENVS
        else:        # Create multiple parallel environments
            env = SubprocVecEnv([make_env(num_rays, model_path, training=training) for _ in range(num_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

 
        
    policy_kwargs = dict(
        net_arch=[128, 128],
        log_std_init=-2.0,
    )
    trainer_name = None
    
    # --- Curriculum Learning Resume ---

    if cl_resume:
        print("Resuming training with Curriculum Learning...")

        pre_trained_path = os.path.join(log_dir, f"{run_id}.zip")

        if os.path.exists(pre_trained_path):
            print(f"✅ Loading pre-trained model from {pre_trained_path}")
            if trainer == "PPO":
                model = PPO.load(pre_trained_path, env=env)
            elif trainer == "SAC":
                model = SAC.load(f"{pre_trained_path}", env=env)
            elif trainer == "TD3":
                model = TD3.load(f"{pre_trained_path}", env=env)
            elif trainer == "TQC":
                model = TQC.load(f"{pre_trained_path}", env=env)
            else:   
                raise ValueError(f"Unsupported trainer: {trainer}. Choose from PPO, SAC, TD3, or TQC.")
            
            normalize_path = os.path.join(log_dir, f"{pre_trained_path}.pkl")
            if os.path.exists(normalize_path):
                env = VecNormalize.load(normalize_path, env)
                print(f"✅ Loaded normalization parameters from {normalize_path}")
            else:
                print(f"⚠️ Normalization file {normalize_path} not found, using default normalization")
        else:
            # BC CASE
            print(f"⚠️ No model named {pre_trained_path} found, loading BC policy from {bc_policy_path}/best_policy.pt")
            checkpoint = torch.load(f"{bc_policy_path}/best_policy.pt", map_location="cpu", weights_only=False)
            
            


            if trainer == "PPO":
                policy_kwargs = dict(
                    net_arch=[128, 128],
                    log_std_init=-2.0,
                )
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    device="cpu"
                )
                model.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
                print("✅ Loaded BC policy successfully into PPO model.")
            elif trainer == "SAC":
                policy_kwargs = dict(
                    net_arch=[128, 128],
                    log_std_init=-2.0,
                )
                model = SAC(
                    policy="MlpPolicy",
                    env=env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    device="cpu"
                )
                model.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
                print("✅ Loaded BC policy successfully into SAC model.")
            elif trainer == "TD3":
                policy_kwargs = dict(
                    net_arch=[128, 128],
                    log_std_init=-2.0,
                )
                model = TD3(
                    policy="MlpPolicy",
                    env=env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    device="cpu"
                )
                model.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
                print("✅ Loaded BC policy successfully into TD3 model.")
            elif trainer == "TQC":
                policy_kwargs = dict(
                    net_arch=(dict(pi=[128, 128], qf=[128, 128])),
                    log_std_init=-2.0,
                    activation_fn=torch.nn.modules.activation.Tanh,
                
                )
                model = TQC(
                    policy="MlpPolicy",
                    env=env,
                    verbose=1,                         # turn on console logs
                    tensorboard_log=log_dir,           # keep TB logging consistent
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    learning_rate=0.0001,
                    batch_size=1024,
                    gamma=0.98,
                    tau=0.005,
                    ent_coef="auto_0.01",
                    policy_kwargs=policy_kwargs,
                )
                checkpoint['policy_state_dict']['actor.latent_pi.0.weight'] = checkpoint['policy_state_dict']['mlp_extractor.policy_net.0.weight']
                checkpoint['policy_state_dict']['actor.latent_pi.0.bias'] = checkpoint['policy_state_dict']['mlp_extractor.policy_net.0.bias']
                checkpoint['policy_state_dict']['actor.latent_pi.2.weight'] = checkpoint['policy_state_dict']['mlp_extractor.policy_net.2.weight']
                checkpoint['policy_state_dict']['actor.latent_pi.2.bias'] = checkpoint['policy_state_dict']['mlp_extractor.policy_net.2.bias']
                checkpoint['policy_state_dict']['actor.mu.weight'] = checkpoint['policy_state_dict']['action_net.weight']
                checkpoint['policy_state_dict']['actor.mu.bias'] = checkpoint['policy_state_dict']['action_net.bias']
                #print(model.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False))
                model.policy._squash_output = False
                print("✅ Loaded BC policy successfully into TQC model.")

            #print("\n\n\n\n", model.policy.__dict__, "\n\n\n\n")
            # Evaluate policy for some steps just to check if it works, with rendering
            
        
            normalize_path = os.path.join(bc_policy_path, f"vec_normalize.pkl")
            if os.path.exists(normalize_path):
                env = VecNormalize.load(normalize_path, env)
                print(f"✅ Loaded normalization parameters from {normalize_path}")
            else:
                print(f"⚠️ Normalization file {normalize_path} not found, using default normalization")
                env = VecNormalize(env, norm_obs=False, norm_reward=False)

            trainer_name = trainer.upper()   # e.g. "TQC"
            if trainer_name is not None:
                print(f"Training {trainer_name} agent with Curriculum Learning for {num_steps} steps...")

   

    # --- Standard RL Training Setup ---

    else:
        if trainer == "PPO" or trainer == "ppo":
            # Create PPO model with built-in logging
            model = PPO(
                policy="MlpPolicy",
                env=env,
                tensorboard_log=log_dir,
                learning_rate=3e-4,
                device = "cpu",
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
            )
            trainer_name = "PPO"
            
        elif trainer == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                learning_rate=0.0001,    # Kept same
                buffer_size=1000000,     # Kept same
                learning_starts=5000,    # Increased from 1000
                batch_size=1024,          # Increased from 256
                tau=0.01,               # Increased from 0.005
                gamma=0.99,             # Kept same
                train_freq=1,           # Kept same
                gradient_steps=1,       # Kept same
                ent_coef="auto",        # Kept same
                policy_kwargs=policy_kwargs
            )
            trainer_name = "SAC"

        elif trainer == "TD3":
            model = TD3(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                learning_rate=0.0003,  # Reduced from 0.001
                buffer_size=1000000,   # Kept same
                learning_starts=20000,  
                batch_size=1024,        # Reduced from 256
                tau=0.005,            # Kept same
                gamma=0.98,           # Slightly reduced from 0.99
                #train_freq=(num_envs*100, "step"),    # Explicit step/epoch setting
                policy_kwargs=dict(
                    net_arch=[256, 256],
                    #noise_std=0.2,
                    #noise_clip=0.5
                )
            )
            trainer_name = "TD3"

        elif trainer == "TQC":
            model = TQC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                learning_rate=0.0001,       # Balanced learning rate
                batch_size=1024,            # Larger batch for stability
                gamma=0.98,                # Good for mid-horizon tasks
                tau=0.005,                 # Default target update rate
                ent_coef="auto_0.01",    # Adaptive entropy coefficient
                #n_quantiles=25,            # Default quantile count
                #top_quantiles_to_drop=2,   # Default truncation
                policy_kwargs=policy_kwargs
            )
            trainer_name = "TQC"

        elif trainer == "A2C":
            model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs
            )
            trainer_name = "A2C"

        elif trainer == "BC":
            print("Loading BC model from bc_policy/bc_model.zip...")

            eval_env = DummyVecEnv([lambda: hamrrln(
                num_rays=num_rays, 
                model_path=model_path, 
                training=False,
                n_humans=5,
                n_stacking=n_stacking,  # Must match IL training
                enable_stacking=stacking,
                render_mode="human",  # For evaluation  
            )])
            
            if os.path.exists("bc_policy/vec_normalize.pkl"):
                eval_env = VecNormalize.load("bc_policy/vec_normalize.pkl", eval_env)
                print("✅ Loaded VecNormalize parameters from IL training")
            else:
                print("⚠️  VecNormalize file not found, using default normalization")
                eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

            eval_env.training = False
            eval_env.norm_reward = False
            
            model = PPO.load("bc_policy/bc_model.zip", env=eval_env)
            print("✅ Successfully loaded BC model")
            
            import pickle
            if os.path.exists("bc_policy/training_config.pkl"):
                with open("bc_policy/training_config.pkl", "rb") as f:
                    config = pickle.load(f)
                print(f"BC model trained on {config['training_samples']} samples")
                print(f"Best validation loss: {config['best_val_loss']:.4f}")
        else:
            raise ValueError(f"Unsupported trainer: {trainer}. Choose from PPO, SAC, TD3, TQC, or BC.")


     
    
    # Only keep necessary callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=2000000,
        save_path="./policy_checkpoints/",
        name_prefix=run_id
    )

    reward_callback = RewardCallback()
    scenario_success_callback = ScenarioSuccessCallback(log_freq=2000, verbose=1)



    callbacks = [checkpoint_callback, reward_callback, scenario_success_callback]
    

    if render_training and num_envs == 1:
        # Add render callback only if training in a single environment
        callbacks.append(RenderCallback())

    # Train the model
    if trainer_name is not None:
        print(f"Training {trainer_name} agent for {num_steps} steps...")

    try:
        model.learn(
            total_timesteps=num_steps,
            callback=callbacks,
            tb_log_name=run_id  # This ensures logs go to the correct subdirectory
        )
    except KeyboardInterrupt: # Handle CTRL+C gracefully
        print("\n⏹  CTRL+C detected — saving intermediate checkpoint…")
        _safe_save(model, env, run_id, log_dir)
        env.close()
        return  # exit train_agent cleanly

    
    # Save the final model
    model.save(f"{run_id}")
    env.save(os.path.join(log_dir, f"{run_id}.pkl"))

    print("\n=== FINAL TRAINING SCENARIO STATISTICS ===")
    scenario_success_callback._log_scenario_summary()

    env.close()










    
if __name__ == "__main__":
    # Optional: force CPU if CUDA/CuDNN errors persist
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser(
        description="Train or evaluate RL agents in HAMR environment."
    )
    parser.add_argument("--num_rays", type=int, default=NUM_RAYS,
                        help="Number of LiDAR rays.")
    parser.add_argument("--model_path", type=str, default="assets/world.xml",
                        help="Path to the MuJoCo XML file.")
    parser.add_argument("--num_envs", type=int, default=36,
                        help="Number of parallel environments.")
    parser.add_argument("--train", action="store_true",
                        help="Train the agent.")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate the agent.")
    parser.add_argument("--num_steps", type=int, default=10_000_000,
                        help="Training steps.")
    parser.add_argument("--run_id", type=str, default="DEFAULT",
                        help="Run ID for logging and saving.")
    parser.add_argument("--trainer", type=str, default="TQC",
                        help="Trainer to use: PPO, SAC, TD3, TQC, or BC.")
    parser.add_argument("--num_obst", type=int, default=51,
                        help="Number of obstacles.")
    parser.add_argument("--no-stacking", action="store_true",
                        help="Disable observation stacking.")
    parser.add_argument("--n_stacking", type=int, default=N_STACKING,
                        help="Number of stacked observations.")
    parser.add_argument("--CL", action="store_true",
                        help="Enable Curriculum Learning.")
    parser.add_argument("--bc_path", type=str, default="bc_policy/",
                        help="Path to the pre-trained BC policy model.")
    parser.add_argument("--render_training", action="store_true",
                        help="Render the training environment (single env only).")

    args = parser.parse_args()
    stacking = not args.no_stacking
    name = args.run_id

    bc_policy_path = args.bc_path   

    # ──────────────────────────────── EVALUATION ────────────────────────────────
    if args.eval:
        # Create ONE evaluation environment (render_mode="human")
        eval_env = DummyVecEnv([lambda: hamrrln(
            num_rays=args.num_rays,
            model_path=args.model_path,
            training=False,
            n_humans=5,
            n_stacking=args.n_stacking,
            enable_stacking=stacking,
            render_mode="human",
        )])

        # For RL trainers other than BC, try to load VecNormalize statistics
        if args.trainer.upper() != "BC":
            vecnorm_path = os.path.join("./TENSORBOARD/", f"{args.run_id}.pkl")
            if os.path.exists(vecnorm_path):
                eval_env = VecNormalize.load(vecnorm_path, eval_env)
                print()
                print(f"✅ Loaded VecNormalize parameters from {vecnorm_path}")
                print()
            else:
                fallback_path = os.path.join("./TENSORBOARD/", "vecnormalize.pkl")
                print(f"⚠️  VecNormalize file '{vecnorm_path}' not found, trying fallback '{fallback_path}'")
                if os.path.exists(fallback_path):
                    eval_env = VecNormalize.load(fallback_path, eval_env)
                else:
                    print("⚠️  No VecNormalize file found, using default normalization")
                    #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
                    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0) # if .pkl file is found, usually this is not needed

            eval_env.training = False
            eval_env.norm_reward = False

        # ── Load model ───────────────────────────────────────────────────────────
        if args.trainer.upper() == "PPO":
            model = PPO.load(name)
        elif args.trainer.upper() == "SAC":
            model = SAC.load(name)
        elif args.trainer.upper() == "TD3":
            model = TD3.load(name)
        elif args.trainer.upper() == "TQC":
            model = TQC.load(name)


        # Testing IL policy
        elif args.trainer.upper() == "BC":
            # ---------- BC model: reuse existing eval_env (don't recreate it) ----
            if not os.path.exists(f"{bc_policy_path}/best_policy.pt"):
                print(f"❌ BC model file '{bc_policy_path}/best_policy.pt' not found!")
                exit(1)
            if not os.path.exists(f"{bc_policy_path}/training_config.pkl"):
                print(f"❌ BC training config file '{bc_policy_path}/training_config.pkl' not found!")
                exit(1)

            with open(f"{bc_policy_path}/training_config.pkl", "rb") as f:
                policy_data = pickle.load(f)
            checkpoint = torch.load(f"{bc_policy_path}/best_policy.pt",
                                    map_location="cpu", weights_only=False)

            # Wrap EXISTING eval_env with VecNormalize for obs normalization
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
            eval_env.training = False

            # Restore normalization statistics if present
            if 'obs_mean' in checkpoint and 'obs_std' in checkpoint:
                eval_env.obs_rms.mean = checkpoint['obs_mean']
                eval_env.obs_rms.var = checkpoint['obs_std'] ** 2
            else:
                print("⚠️  No normalization stats found in checkpoint, using defaults")

            # Apply angle mask if available
            if 'angle_mask' in checkpoint:
                angle_mask = np.array(checkpoint['angle_mask'])
                eval_env.obs_rms.mean[angle_mask] = 0
                eval_env.obs_rms.var[angle_mask] = 1

            policy_kwargs = dict(
                net_arch=[128, 128],
                log_std_init=-2.0,
            )

            model = PPO(
                policy="MlpPolicy",
                env=eval_env,
                policy_kwargs=policy_kwargs,
                device="cpu"
            )
            model.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
            print("✅ Loaded BC policy successfully.")

        else:
            print(f"❌ Unsupported trainer: {args.trainer}")
            exit(1)

        # ── Manual evaluation loop ───────────────────────────────────────────────
        env = eval_env.envs[0]           # Unwrap the single DummyVecEnv
        env.set_real_time_factor(10)
        N_EVAL_EPISODES = 200

        print("🎥 Starting evaluation with rendering...")
        # for _ in range(N_EVAL_EPISODES):
        #     obs = env.reset()[0]
        #     done = False
        #     while not done:
        #         action, _ = model.predict(obs, deterministic=True)
        #         obs, reward, terminated, truncated, info = env.step(action)
        #         done = terminated or truncated
        #         env.render()  # refresh MuJoCo viewer

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=300, deterministic=True, render=True)


        eval_env.close()

    # ──────────────────────────────── TRAINING ──────────────────────────────────
    if args.train:
        train_agent(
            args.num_rays,
            args.model_path,
            args.num_envs,
            args.num_steps,
            args.run_id,
            training=True,
            trainer=args.trainer,
            stacking=stacking,
            n_stacking=args.n_stacking,
            cl_resume=args.CL,
            bc_policy_path=bc_policy_path,
            render_training=args.render_training
        )
