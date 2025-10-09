from __future__ import annotations
import os
import sys
import argparse
import time
import math
from pathlib import Path
import xml.etree.ElementTree as ET  # (non usato direttamente ma lasciato per compat)
import logging
import pickle

import numpy as np
import torch
from torch import nn

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO, SAC, TD3, A2C
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# ───────────────────────────────────────────────────────────────────────────────
# Path bootstrap: consente import “from core.*” e “from RL.*” eseguendo da root
# ───────────────────────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR   = _THIS_FILE.parents[1]   # <ROOT>/src
_ROOT_DIR  = _THIS_FILE.parents[2]   # <ROOT>

for p in (str(_ROOT_DIR), str(_SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ───────────────────────────────────────────────────────────────────────────────
# Import interni alla nuova struttura
# ───────────────────────────────────────────────────────────────────────────────
from core.lightHAMRRLN import light_hamrrln as hamrrln
from core.env_config import NUM_RAYS, N_STACKING, N_HUMANS
from RL.CNN_lstmNN import CNNLSTMExtractor

from assets.custom_callback import RewardCallback, GPUStatsCallback
from assets.train_classes import (
    ScenarioSuccessCallback,
    PolicySaveCallback,
    RenderCallback,
    AveragedRawReturnByStep,
)

# ───────────────────────────────────────────────────────────────────────────────
# Env vars “safe defaults”
# ───────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTORCH_NUM_THREADS"]:
    os.environ.setdefault(v, "1")

# ============
# Logging Setup
# ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _guard_unique_run_id(
    run_id: str,
    log_dir: str = "logs/TENSORBOARD",
    ckpt_dir: str = "logs/policy_checkpoints",
):
    """
    Blocca se esistono già artefatti per lo stesso run_id.
    """
    candidates = [
        f"{run_id}.zip",
        os.path.join(log_dir, f"{run_id}.pkl"),
    ]
    exists = any(os.path.exists(p) for p in candidates)

    if os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            if fn.startswith(run_id + "_") and fn.endswith(".zip"):
                exists = True
                break

    tb_subdir = os.path.join(log_dir, run_id)
    if os.path.isdir(tb_subdir):
        exists = True

    if exists:
        raise FileExistsError(
            f"\n\nrun_id '{run_id}' già esistente: trovati artefatti. "
            f"Scegli un run_id diverso oppure archivia/rimuovi i file associati."
        )


def _safe_save(model, env, run_id, log_dir, suffix="interrupt"):
    """
    Salva checkpoint + VecNormalize senza sovrascrivere il finale.
    """
    timesteps = getattr(model, "num_timesteps", "0")
    stamp = f"{suffix}_{timesteps}"
    model_path = f"{run_id}_{stamp}.zip"
    norm_path = os.path.join(log_dir, f"{run_id}_{stamp}.pkl")

    model.save(model_path)
    if isinstance(env, VecNormalize):
        env.save(norm_path)

    print(f"💾 Saved model to {model_path}")
    if isinstance(env, VecNormalize):
        print(f"💾 Saved VecNormalize to {norm_path}")


# ───────────────────────────────────────────────────────────────────────────────
# Env factory
# ───────────────────────────────────────────────────────────────────────────────
def make_env(
    num_rays,
    model_path="assets/world.xml",
    training=True,
    n_humans=N_HUMANS,
    render_mode=None,
    n_stacking=10,
    stacking=True,
    max_steps=1000,
):
    def _init():
        env = hamrrln(
            num_rays=num_rays,
            model_path=model_path,
            training=training,
            n_humans=n_humans,
            n_stacking=n_stacking,
            enable_stacking=stacking,
            render_mode=render_mode,
        )
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env

    return _init


# ───────────────────────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────────────────────
def train_agent(
    num_rays,
    model_path="assets/world.xml",
    num_envs=16,
    num_steps=100_000,
    run_id="training1",
    training=True,
    trainer="ppo",
    stacking=True,
    n_stacking=10,
    cl_resume=False,
    init_from: str | None = None,
    init_keep_timesteps: bool = False,
    bc_policy_path="bc_policy/",
    render_training=False,
    force=False,
):
    n_humans = N_HUMANS

    log_dir = "logs/TENSORBOARD"
    ckpt_dir = "logs/policy_checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if not force:
        _guard_unique_run_id(run_id, log_dir=log_dir, ckpt_dir=ckpt_dir)

    # ===== Env =====
    if not training:
        env = hamrrln(
            num_rays=num_rays,
            model_path=model_path,
            training=False,
            n_humans=n_humans,
            n_stacking=n_stacking,
            enable_stacking=stacking,
        )
    else:
        if render_training and num_envs == 1:
            raw_env = DummyVecEnv(
                [
                    lambda: make_env(
                        num_rays,
                        model_path,
                        training=True,
                        n_humans=n_humans,
                        n_stacking=n_stacking,
                        stacking=stacking,
                        render_mode="human",
                    )()
                ]
            )
        else:
            raw_env = SubprocVecEnv(
                [make_env(num_rays, model_path, training=True) for _ in range(num_envs)],
                start_method="forkserver",
            )

        # VecNormalize
        if cl_resume:
            normalize_path = os.path.join(log_dir, f"{run_id}.pkl")
            if os.path.exists(normalize_path):
                env = VecNormalize.load(normalize_path, raw_env)
                print(f"✅ Loaded VecNormalize from {normalize_path}")
            else:
                print(f"⚠️ VecNormalize not found: {normalize_path} → creating fresh stats")
                env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        elif init_from:
            src_norm = os.path.join(log_dir, f"{init_from}.pkl")
            if os.path.exists(src_norm):
                env = VecNormalize.load(src_norm, raw_env)
                print(f"✅ INIT-FROM: loaded VecNormalize from {src_norm}")
            else:
                print(f"⚠️ INIT-FROM: {src_norm} not found → creating fresh stats")
                env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ===== Models =====
    policy_kwargs = dict(net_arch=[128, 128], log_std_init=-2.0)
    trainer_name = None
    model = None

    # ---- Curriculum resume / BC warm-start ----
    if cl_resume:
        print("Resuming training with Curriculum Learning…")
        pre_trained_path = f"{run_id}.zip"

        if os.path.exists(pre_trained_path):
            print(f"✅ Loading pre-trained model from {pre_trained_path}")
            if trainer.upper() == "PPO":
                model = PPO.load(pre_trained_path, env=env, device="auto")
            elif trainer.upper() == "SAC":
                model = SAC.load(pre_trained_path, env=env, device="auto")
            elif trainer.upper() == "TD3":
                model = TD3.load(pre_trained_path, env=env, device="auto")
            elif trainer.upper() == "TQC":
                model = TQC.load(pre_trained_path, env=env, device="auto")
            else:
                raise ValueError("Unsupported trainer for CL resume.")
        else:
            # Warm start con BC
            print(f"⚠️ No '{pre_trained_path}' found, loading BC policy from {bc_policy_path}/best_policy.pt")
            checkpoint = torch.load(f"{bc_policy_path}/best_policy.pt", map_location="cpu", weights_only=False)

            if trainer.upper() == "PPO":
                model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cpu")
                model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
            elif trainer.upper() == "SAC":
                model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cpu")
                model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
            elif trainer.upper() == "TD3":
                model = TD3("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cpu")
                model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
            elif trainer.upper() == "TQC":
                policy_kwargs = dict(
                    net_arch=dict(pi=[128, 128], qf=[128, 128]),
                    log_std_init=-2.0,
                    activation_fn=torch.nn.Tanh,
                )
                model = TQC(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    learning_rate=1e-4,
                    batch_size=1024,
                    gamma=0.98,
                    tau=0.005,
                    ent_coef="auto_0.01",
                    policy_kwargs=policy_kwargs,
                )
                # Rimappa i nomi dei layer dal checkpoint BC → TQC actor
                sd = checkpoint["policy_state_dict"]
                sd["actor.latent_pi.0.weight"] = sd["mlp_extractor.policy_net.0.weight"]
                sd["actor.latent_pi.0.bias"]   = sd["mlp_extractor.policy_net.0.bias"]
                sd["actor.latent_pi.2.weight"] = sd["mlp_extractor.policy_net.2.weight"]
                sd["actor.latent_pi.2.bias"]   = sd["mlp_extractor.policy_net.2.bias"]
                sd["actor.mu.weight"]          = sd["action_net.weight"]
                sd["actor.mu.bias"]            = sd["action_net.bias"]
                model.policy._squash_output = False
            else:
                raise ValueError("Unsupported trainer for BC warm-start.")

            # VecNormalize da BC se disponibile
            normalize_path = os.path.join(bc_policy_path, "vec_normalize.pkl")
            if os.path.exists(normalize_path):
                env = VecNormalize.load(normalize_path, env)
                print(f"✅ Loaded normalization parameters from {normalize_path}")
            else:
                print(f"⚠️ Normalization file {normalize_path} not found, using default normalization")
                env = VecNormalize(env, norm_obs=False, norm_reward=False)

            trainer_name = trainer.upper()
            if trainer_name:
                print(f"Training {trainer_name} agent with Curriculum Learning for {num_steps} steps…")

    elif init_from:
        print(f"INIT-FROM: cloning weights from '{init_from}' into NEW run_id '{run_id}' …")
        src_model = f"{init_from}.zip"
        if not os.path.exists(src_model):
            raise FileNotFoundError(f"INIT-FROM: source model '{src_model}' not found.")

        if trainer.upper() == "PPO":
            model = PPO.load(src_model, env=env, device="auto")
        elif trainer.upper() == "SAC":
            model = SAC.load(src_model, env=env, device="auto")
        elif trainer.upper() == "TD3":
            model = TD3.load(src_model, env=env, device="auto")
        elif trainer.upper() == "TQC":
            model = TQC.load(src_model, env=env, device="auto")
        else:
            raise ValueError("Unsupported trainer.")
        trainer_name = trainer.upper()
        print("✅ INIT-FROM: weights loaded. Starting a NEW run; the source files remain untouched.")

    else:
        # ===== Standard training setup =====
        if trainer.lower() == "ppo":
            def linear_schedule(initial_value: float):
                return lambda progress_remaining: progress_remaining * initial_value

            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                ortho_init=True,
                log_std_init=-1.0,
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule(3e-4),
                n_steps=640,
                batch_size=512,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.10,
                ent_coef=1e-3,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.015,
                use_sde=True,
                sde_sample_freq=64,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir,
                device="cpu",
                verbose=1,
            )
            trainer_name = "PPO"

        elif trainer.lower() == "sac":
            model = SAC(
                "MlpPolicy",
                env,
                tensorboard_log=log_dir,
                learning_rate=3e-4,
                buffer_size=int(1e6),
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                target_update_interval=1,
                policy_kwargs={"net_arch": [256, 256], "log_std_init": -2},
                verbose=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            trainer_name = "SAC"

        elif trainer.lower() == "td3":
            from stable_baselines3.common.noise import NormalActionNoise

            use_cuda = torch.cuda.is_available()
            batch_size = 1024 if use_cuda else 512

            action_noise = NormalActionNoise(
                mean=np.array([0.0, 0.0], dtype=np.float32),
                sigma=np.array([0.08, 0.25], dtype=np.float32),
            )

            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
                activation_fn=nn.ReLU,
            )
            model = TD3(
                "MlpPolicy",
                env,
                tensorboard_log=log_dir,
                device=("cuda" if use_cuda else "cpu"),
                learning_rate=3e-4,
                buffer_size=2_000_000,
                batch_size=batch_size,
                gamma=0.99,
                tau=0.005,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_delay=2,
                action_noise=action_noise,
                learning_starts=20_000,
                verbose=1,
                policy_kwargs=policy_kwargs,
            )
            trainer_name = "TD3"

        elif trainer.lower() == "tqc":
            use_cuda = torch.cuda.is_available()
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
                activation_fn=nn.ReLU,
                n_critics=2,
                n_quantiles=25,
            )
            buffer_size = 300_000
            batch_size = 512 if use_cuda else 256

            model = TQC(
                "MlpPolicy",
                env,
                tensorboard_log=log_dir,
                device=("cuda" if use_cuda else "cpu"),
                learning_rate=3e-4,
                buffer_size=buffer_size,
                batch_size=batch_size,
                train_freq=(1, "step"),
                gradient_steps=1,
                learning_starts=20_000,
                tau=0.005,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
            trainer_name = "TQC"

        elif trainer.upper() == "A2C":
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
                policy_kwargs=policy_kwargs,
            )
            trainer_name = "A2C"

        elif trainer.upper() == "BC":
            print("Loading BC model from bc_policy/bc_model.zip…")
            eval_env = DummyVecEnv(
                [
                    lambda: hamrrln(
                        num_rays=num_rays,
                        model_path=model_path,
                        training=False,
                        n_humans=N_HUMANS,
                        n_stacking=n_stacking,
                        enable_stacking=stacking,
                        render_mode="human",
                    )
                ]
            )

            if os.path.exists("bc_policy/vec_normalize.pkl"):
                eval_env = VecNormalize.load("bc_policy/vec_normalize.pkl", eval_env)
                print("✅ Loaded VecNormalize parameters from IL training")
            else:
                print("⚠️ VecNormalize file not found, using default normalization")
                eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

            eval_env.training = False
            eval_env.norm_reward = False

            model = PPO.load("bc_policy/bc_model.zip", env=eval_env)
            print("✅ Successfully loaded BC model")

            if os.path.exists("bc_policy/training_config.pkl"):
                with open("bc_policy/training_config.pkl", "rb") as f:
                    config = pickle.load(f)
                print(f"BC model trained on {config.get('training_samples','?')} samples")
                print(f"Best validation loss: {config.get('best_val_loss', float('nan')):.4f}")
            trainer_name = "BC"

        else:
            raise ValueError("Unsupported trainer: choose among PPO, SAC, TD3, TQC, A2C, BC.")

    # Ensure VecNormalize flags
    if isinstance(env, VecNormalize):
        env.training = True
        env.norm_reward = True

    print("\n\nGPU disponibile:", torch.cuda.is_available())
    print("Device policy SB3:", getattr(model.policy, "device", "N/A"))
    print("Param device:", next(model.policy.parameters()).device, "\n\n")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=2_000_000,
        save_path=ckpt_dir,
        name_prefix=run_id,
    )
    reward_callback = RewardCallback()
    scenario_success_callback = ScenarioSuccessCallback(log_freq=2000, verbose=1)
    avg_raw_cb = AveragedRawReturnByStep(
        key_return="raw_episode_return",
        window=100,
        ema_alpha=0.1,
        log_prefix="rollout",
    )

    callbacks = [
        checkpoint_callback,
        reward_callback,
        scenario_success_callback,
        GPUStatsCallback(gpu_index=0, min_interval_s=10),
        avg_raw_cb,
    ]
    if render_training and num_envs == 1:
        callbacks.append(RenderCallback())

    if trainer_name:
        print(f"Training {trainer_name} agent for {num_steps} steps…")

    try:
        model.learn(
            total_timesteps=num_steps,
            reset_num_timesteps=(not cl_resume and not init_keep_timesteps),
            callback=callbacks,
            tb_log_name=run_id,
        )
    except KeyboardInterrupt:
        print("\n⏹  CTRL+C detected — saving intermediate checkpoint…")
        _safe_save(model, env, run_id, log_dir)
        env.close()
        return

    # Final save
    model.save(f"{run_id}")
    if isinstance(env, VecNormalize):
        env.save(os.path.join(log_dir, f"{run_id}.pkl"))

    print("\n=== FINAL TRAINING SCENARIO STATISTICS ===")
    scenario_success_callback._log_scenario_summary()

    env.close()


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate RL agents in HAMR environment.")
    parser.add_argument("--num_rays", type=int, default=NUM_RAYS, help="Number of LiDAR rays.")
    parser.add_argument("--model_path", type=str, default="assets/world.xml", help="Path to the MuJoCo XML file.")
    parser.add_argument("--num_envs", type=int, default=36, help="Number of parallel environments.")
    parser.add_argument("--train", action="store_true", help="Train the agent.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the agent.")
    parser.add_argument("--num_steps", type=int, default=10_000_000, help="Training steps.")
    parser.add_argument("--run_id", type=str, default="DEFAULT", help="Run ID for logging and saving.")
    parser.add_argument("--trainer", type=str, default="TQC", help="Trainer: PPO, SAC, TD3, TQC, A2C, or BC.")
    parser.add_argument("--num_obst", type=int, default=51, help="(legacy) Number of obstacles.")
    parser.add_argument("--no-stacking", action="true", help="Disable observation stacking.")
    parser.add_argument("--n_stacking", type=int, default=N_STACKING, help="Number of stacked observations.")
    parser.add_argument("--CL", action="store_true", help="Enable Curriculum Learning.")
    parser.add_argument("--bc_path", type=str, default="bc_policy", help="Path to the pre-trained BC policy model.")
    parser.add_argument("--render_training", action="store_true", help="Render the training environment (single env only).")
    parser.add_argument("--force", action="store_true", help="Force overwrite if run_id exists.")
    parser.add_argument("--init_from", type=str, default=None, help="Clone weights from this run_id into NEW run_id.")
    parser.add_argument("--init_keep_timesteps", action="store_true", help="With --init_from, keep timesteps (no reset).")

    args = parser.parse_args()
    stacking = not args.no_stacking
    name = args.run_id
    bc_policy_path = args.bc_path

    # ──────────────────────────────── EVALUATION ────────────────────────────────
    if args.eval:
        eval_env = DummyVecEnv(
            [
                lambda: hamrrln(
                    num_rays=args.num_rays,
                    model_path=args.model_path,
                    training=False,
                    n_humans=N_HUMANS,
                    n_stacking=args.n_stacking,
                    enable_stacking=stacking,
                    render_mode="human",
                )
            ]
        )

        if args.trainer.upper() != "BC":
            vecnorm_path = os.path.join("logs/TENSORBOARD", f"{args.run_id}.pkl")
            if os.path.exists(vecnorm_path):
                eval_env = VecNormalize.load(vecnorm_path, eval_env)
                print(f"\n✅ Loaded VecNormalize parameters from {vecnorm_path}\n")
            else:
                fallback_path = os.path.join("logs/TENSORBOARD", "vecnormalize.pkl")
                print(f"⚠️ VecNormalize '{vecnorm_path}' not found, trying fallback '{fallback_path}'")
                if os.path.exists(fallback_path):
                    eval_env = VecNormalize.load(fallback_path, eval_env)
                else:
                    print("⚠️ No VecNormalize file found, using default normalization")
                    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            eval_env.training = False
            eval_env.norm_reward = False

        # Load model
        if args.trainer.upper() == "PPO":
            model = PPO.load(name, env=eval_env)
        elif args.trainer.upper() == "SAC":
            model = SAC.load(name, env=eval_env)
        elif args.trainer.upper() == "TD3":
            model = TD3.load(name, env=eval_env)
        elif args.trainer.upper() == "TQC":
            model = TQC.load(name, env=eval_env)
        elif args.trainer.upper() == "BC":
            if not os.path.exists(f"{bc_policy_path}/best_policy.pt"):
                print(f"❌ BC model file '{bc_policy_path}/best_policy.pt' not found!")
                sys.exit(1)
            if not os.path.exists(f"{bc_policy_path}/vec_normalize.pkl"):
                print(f"❌ BC VecNormalize file '{bc_policy_path}/vec_normalize.pkl' not found!")
                sys.exit(1)

            with open(f"{bc_policy_path}/vec_normalize.pkl", "rb") as f:
                _ = pickle.load(f)
            checkpoint = torch.load(f"{bc_policy_path}/best_policy.pt", map_location="cpu", weights_only=False)

            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
            eval_env.training = False

            if "obs_mean" in checkpoint and "obs_std" in checkpoint:
                eval_env.obs_rms.mean = checkpoint["obs_mean"]
                eval_env.obs_rms.var = checkpoint["obs_std"] ** 2
            if "angle_mask" in checkpoint:
                angle_mask = np.array(checkpoint["angle_mask"])
                eval_env.obs_rms.mean[angle_mask] = 0
                eval_env.obs_rms.var[angle_mask] = 1

            policy_kwargs = dict(net_arch=[128, 128], log_std_init=-2.0)
            model = PPO("MlpPolicy", eval_env, policy_kwargs=policy_kwargs, device="cpu")
            state_dict = checkpoint["policy_state_dict"] if isinstance(checkpoint, dict) else checkpoint
            model.policy.load_state_dict(state_dict, strict=False)
            print("✅ Loaded BC policy successfully.")
        else:
            print(f"❌ Unsupported trainer: {args.trainer}")
            sys.exit(1)

        # Esecuzione valutazione
        env = eval_env.envs[0]
        if hasattr(env, "set_real_time_factor"):
            env.set_real_time_factor(10)
        try:
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=300, deterministic=True, render=True)
        except KeyboardInterrupt:
            print("\n⏹  CTRL+C detected — evaluation interrupted.")
            eval_env.close()
            sys.exit(0)
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
            init_from=args.init_from,
            init_keep_timesteps=args.init_keep_timesteps,
            bc_policy_path=bc_policy_path,
            render_training=args.render_training,
            force=args.force,
        )
