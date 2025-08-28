#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate social performance metrics for trained RL models in HAMRRLN.

Metrics per episode:
- result (success/collision/timeout)
- steps, time_s
- path_len, min_dist_human
- start_to_goal (L* = straight-line)
- SPL
- space_compliance (fraction of steps with min human distance >= SAFE_DIST)
- scenario_id (from env info)

Aggregates:
- Per-scenario metrics (grouped by scenario_id)
- Overall metrics per model

Outputs:
- social_metrics_<run_id>_<TRAINER>.csv               (per-episode)
- social_metrics_summary_<run_id>_<TRAINER>.csv       (overall, one row)
- social_metrics_by_scenario_<run_id>_<TRAINER>.csv   (per-scenario)

Usage example:
    python3 evaluate_social_metrics.py \
        --run_id Vediamo \
        --trainer TQC \
        --episodes 1000 \
        --model_path ./MODELS/Vediamo/Vediamo \
        --xml assets/world.xml \
        --safe_dist 0.6
"""
import os
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC, TD3, A2C
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed

from HAMRRLN import hamrrln, N_HUMANS
from IL_HAMRRLN import NUM_RAYS, N_STACKING
import matplotlib.pyplot as plt



@dataclass
class EpisodeMetrics:
    episode: int
    scenario_id: int
    result: str
    steps: int
    time_s: float
    path_len: float
    min_dist_human: float
    start_to_goal: float
    spl: float
    space_compliance: float
    avg_angular_jerk: float 
    


def load_eval_env(run_id: str, xml: str, stacking: bool, n_stacking: int,
                  n_envs: int, seed: int) -> VecNormalize:
    base_vec = build_vec_env(n_envs, xml, stacking, n_stacking, seed)
    vecnorm_path = os.path.join("./TENSORBOARD", f"{run_id}.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, base_vec)
        print(f"✅ Loaded VecNormalize parameters from {vecnorm_path}")
    else:
        env = VecNormalize(base_vec, norm_obs=False, norm_reward=False)
        print("ℹ️ No VecNormalize stats found — proceeding without normalization.")
    env.training = False
    env.norm_reward = False
    return env



def load_model(trainer: str, run_id: str, env: VecNormalize, bc_dir: str):
    t = trainer.upper()
    if t != "BC":
        # Try common locations (and .zip variants)
        candidates = [
            run_id,                                      # user may pass a full path
            f"./MODELS/{run_id}/{run_id}",
            f"./models/{run_id}",
        ]
        model_path = None
        for p in candidates:
            if os.path.exists(p) or os.path.exists(p + ".zip"):
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError(
                f"Model not found for run_id='{run_id}'. Tried: {candidates} (+ '.zip'). "
                "Either pass a full path as run_id or re-enable --model_path."
            )
        loader = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "TQC": TQC, "A2C": A2C}.get(t)
        return loader.load(model_path, env=env)

    # --- BC branch unchanged ---
    import torch, pickle
    ckpt_path = os.path.join(bc_dir, "best_policy.pt")
    config_pkl = os.path.join(bc_dir, "training_config.pkl")
    if not os.path.exists(ckpt_path) or not os.path.exists(config_pkl):
        raise FileNotFoundError("BC files best_policy.pt or training_config.pkl not found in bc_policy/")
    with open(config_pkl, "rb") as f:
        _ = pickle.load(f)
    policy_kwargs = dict(net_arch=[128, 128], log_std_init=-2.0)
    model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs, device="cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
    print("✅ Loaded BC policy into PPO container for evaluation.")
    return model



def episode_rollout(env: VecNormalize, model, safe_dist: float) -> EpisodeMetrics:
    # Reset the vectorized env (shape: [n_envs, obs_dim]) with n_envs=1
    obs = env.reset()
    base_env = env.venv.envs[0]  # unwrap once for attributes

    prev_pos = np.array(base_env.robot_pos, dtype=np.float32)
    start_pos = prev_pos.copy()
    goal_pos = np.array(base_env.target_pos[:2], dtype=np.float32)
    L_star = float(np.linalg.norm(goal_pos - start_pos))

    path_len = 0.0
    min_dist_human = float("inf")
    steps = 0
    space_ok_steps = 0
    t0 = time.time()
    last_result = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        steps += 1
        cur_pos = np.array(base_env.robot_pos, dtype=np.float32)
        path_len += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos

        humans_xy = base_env.humans_state_numpy[:, :2] if base_env.humans_state_numpy.size > 0 else np.empty((0, 2))
        if humans_xy.size > 0:
            step_min = float(np.min(np.linalg.norm(humans_xy - cur_pos[None, :], axis=1)))
            min_dist_human = min(min_dist_human, step_min)
            if step_min >= safe_dist:
                space_ok_steps += 1
        else:
            space_ok_steps += 1

        if bool(dones[0]):  # vec env returns arrays
            last_result = infos[0].get("episode_result", None)
            break

    elapsed = time.time() - t0
    if last_result is None:
        last_result = "success" if np.linalg.norm(base_env.robot_pos[:2] - goal_pos) < 0.5 else "timeout"

    success_flag = 1.0 if last_result == "success" else 0.0
    denom = max(path_len, L_star, 1e-6)
    spl = success_flag * (L_star / denom)
    space_compliance = space_ok_steps / max(steps, 1)

    return EpisodeMetrics(
        episode=0,
        scenario_id = -1,
        result=last_result,
        steps=steps,
        time_s=float(elapsed),
        path_len=path_len,
        min_dist_human=(min_dist_human if min_dist_human != float("inf") else np.nan),
        start_to_goal=L_star,
        spl=spl,
        space_compliance=space_compliance,
        avg_angular_jerk=float('nan')  # Placeholder, to be computed in parallel evaluation
    )



def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    # Overall (single-row) summary
    total = len(df)
    s = {
        "episodes": total,
        "success_rate": (df["result"] == "success").mean(),
        "collision_rate": (df["result"] == "collision").mean(),
        "timeout_rate": (df["result"] == "timeout").mean(),
        "avg_path_len_success": df.loc[df["result"] == "success", "path_len"].mean(),
        "avg_min_dist_human": df["min_dist_human"].mean(),
        "mean_SPL": df["spl"].mean(),
        "mean_space_compliance": df["space_compliance"].mean(),
        "mean_avg_angular_jerk": df["avg_angular_jerk"].mean(),
    }
    return pd.DataFrame([s])


def summarize_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    # Per-scenario summary
    def agg_fun(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "episodes": len(g),
            "success_rate": (g["result"] == "success").mean(),
            "collision_rate": (g["result"] == "collision").mean(),
            "timeout_rate": (g["result"] == "timeout").mean(),
            "avg_path_len_success": g.loc[g["result"] == "success", "path_len"].mean(),
            "avg_min_dist_human": g["min_dist_human"].mean(),
            "mean_SPL": g["spl"].mean(),
            "mean_space_compliance": g["space_compliance"].mean(),
            "mean_avg_angular_jerk": g["avg_angular_jerk"].mean(),
        })
    out = df.groupby("scenario_id", dropna=False).apply(agg_fun).reset_index()
    out = out.sort_values("scenario_id", ascending=True)
    return out

def make_env_fn(xml: str, stacking: bool, n_stacking: int, seed: int | None):
    def _init():
        env = hamrrln(
            num_rays=NUM_RAYS,
            model_path=xml,
            training=False,
            n_humans=N_HUMANS,
            n_stacking=n_stacking,
            enable_stacking=stacking,
            render_mode=None,
        )
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def build_vec_env(n_envs: int, xml: str, stacking: bool, n_stacking: int, seed: int) -> VecEnv:
    """Create DummyVecEnv (1) or SubprocVecEnv (>=2)."""
    if n_envs <= 1:
        return DummyVecEnv([make_env_fn(xml, stacking, n_stacking, seed)])
    set_random_seed(seed)
    env_fns = [make_env_fn(xml, stacking, n_stacking, seed + i) for i in range(n_envs)]
    # start_method="fork" is fastest on Linux
    return SubprocVecEnv(env_fns, start_method="fork")

def evaluate_parallel(env: VecNormalize, model, episodes_target: int, safe_dist: float, dt: float):
    n_envs = env.num_envs
    obs = env.reset()
    base = env.venv  # SubprocVecEnv or DummyVecEnv

    def get_attr(name): return base.get_attr(name)
    def env_method(name, indices=None): return base.env_method(name, indices=indices)

    # Initial baselines
    # Refresh goal from attribute (no remote method calls)
    goals = np.array([np.array(get_attr("target_pos")[i][:2], dtype=np.float32) for i in range(n_envs)], dtype=np.float32)

    last_xy = np.array([p[:2] for p in get_attr("robot_pos")], dtype=np.float32)
    L_star = np.linalg.norm(goals - last_xy, axis=1)

    try:
        infos0 = env_method("_get_info")
        scenario_ids = np.array([int(info.get("scenario_id", -1)) if isinstance(info, dict) else -1 for info in infos0], dtype=int)
    except Exception:
        scenario_ids = np.full(n_envs, -1, dtype=int)

    path_len = np.zeros(n_envs, dtype=np.float64)
    min_dist = np.full(n_envs, np.inf, dtype=np.float64)
    steps    = np.zeros(n_envs, dtype=int)
    space_ok = np.zeros(n_envs, dtype=int)
    t0       = np.array([time.time()] * n_envs, dtype=np.float64)

    records = []
    completed = 0
    ep_counter = 0

    yaw_history = [[] for _ in range(n_envs)]  # For angular jerk calculation
    # pos_history = [[] for _ in range(n_envs)]  # For linear jerk
    angular_vel_history = [[] for _ in range(n_envs)]  # For jerk calculation

    while completed < episodes_target:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        robot_states = get_attr("robot_pos")   # expect [x, y, yaw] or [x, y, yaw, ...]
        humans_list  = get_attr("humans_state_numpy")

        for i in range(n_envs):
            cur_xy = np.array(robot_states[i][:2], dtype=np.float32)
            # --- path length ---
            path_len[i] += float(np.linalg.norm(cur_xy - last_xy[i]))
            last_xy[i] = cur_xy

            # --- yaw angle history (assume index 2 is yaw angle) ---
            yaw = float(robot_states[i][2]) if len(robot_states[i]) >= 3 else 0.0
            yaw_history[i].append(yaw)

            # --- angular velocity from unwrapped yaw ---
            if len(yaw_history[i]) >= 2:
                yaw_unwrapped = np.unwrap(np.array(yaw_history[i], dtype=np.float64))
                ang_vel = (yaw_unwrapped[-1] - yaw_unwrapped[-2]) / dt  # rad/s
                angular_vel_history[i].append(float(ang_vel))
            else:
                angular_vel_history[i].append(0.0)


            # --- human distances + space compliance ---
            h = humans_list[i]
            if h is not None and getattr(h, "size", 0) > 0:
                step_min = float(np.min(np.linalg.norm(h[:, :2] - cur_xy[None, :], axis=1)))
                if step_min >= safe_dist:
                    space_ok[i] += 1
                if step_min < min_dist[i]:
                    min_dist[i] = step_min
            else:
                space_ok[i] += 1

            steps[i] += 1

            # steps[i] += 1

        for i in range(n_envs):
            if dones[i]:
                ep_counter += 1
                result = infos[i].get("episode_result", None)
                if result is None:
                    result = "success" if np.linalg.norm(last_xy[i] - goals[i]) < 0.5 else "timeout"

                elapsed = time.time() - t0[i]
                success_flag = 1.0 if result == "success" else 0.0
                denom = max(path_len[i], L_star[i], 1e-6)
                spl = success_flag * (L_star[i] / denom)
                space_comp = space_ok[i] / max(steps[i], 1)
                md = min_dist[i] if np.isfinite(min_dist[i]) else np.nan

                #                 # --- Jerk calculation ---
                # positions = np.array(pos_history[i])  # shape (steps, 2)
                # if len(positions) >= 3:
                #     # Compute velocities (Δpos per step)
                #     vels = np.diff(positions, axis=0)
                #     # Compute accelerations (Δvel per step)
                #     accs = np.diff(vels, axis=0)
                #     # Compute jerk magnitudes (Δacc per step)
                #     jerks = np.diff(accs, axis=0)
                #     jerk_magnitudes = np.linalg.norm(jerks, axis=1)
                #     avg_jerk = float(np.mean(jerk_magnitudes))
                # else:
                #     avg_jerk = float('nan')


                ang_vels = np.array(angular_vel_history[i], dtype=np.float64)
                if len(ang_vels) >= 3:
                    ang_acc  = np.diff(ang_vels) / dt
                    # angular jerk in rad/s^3
                    ang_jerk = np.diff(ang_acc) / dt
                    avg_ang_jerk = float(np.mean(np.abs(ang_jerk)))
                else:
                    avg_ang_jerk = float('nan')





                records.append({
                    "episode": ep_counter,
                    "scenario_id": int(scenario_ids[i]),
                    "result": str(result),
                    "steps": int(steps[i]),
                    "time_s": float(elapsed),
                    "path_len": float(path_len[i]),
                    "min_dist_human": float(md),
                    "start_to_goal": float(L_star[i]),
                    "spl": float(spl),
                    "space_compliance": space_ok[i] / max(steps[i], 1),
                    "avg_angular_jerk": avg_ang_jerk
                })
                completed += 1
                if completed % max(1, episodes_target // 20) == 0:
                    print(f"… {completed}/{episodes_target} episodes done")
                if completed >= episodes_target:
                    break

                # Refresh baselines for auto-reset slot i
                goals[i] = np.array(get_attr("target_pos")[i][:2], dtype=np.float32)
                last_xy[i] = np.array(get_attr("robot_pos")[i][:2], dtype=np.float32)
                L_star[i]  = float(np.linalg.norm(goals[i] - last_xy[i]))
                try:
                    new_info = env_method("_get_info", indices=i)[0]
                    scenario_ids[i] = int(new_info.get("scenario_id", -1)) if isinstance(new_info, dict) else -1
                except Exception:
                    scenario_ids[i] = -1

                path_len[i] = 0.0
                min_dist[i] = np.inf
                steps[i]    = 0
                space_ok[i] = 0
                t0[i]       = time.time()

                yaw_history[i] = []
                # pos_history[i] = []
                angular_vel_history[i] = []

    return records




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Run/model ID (used for VecNormalize lookup).")
    parser.add_argument("--trainer", type=str, required=True, help="PPO, SAC, TD3, TQC, A2C, or BC")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes.")
    #parser.add_argument("--model_path", type=str, required=False, help="Path to model zip for RL trainers.")
    parser.add_argument("--bc_dir", type=str, default="./bc_policy", help="Dir containing BC artifacts.")
    parser.add_argument("--xml", type=str, default="assets/world.xml", help="MuJoCo XML path.")
    parser.add_argument("--safe_dist", type=float, default=0.6, help="Personal-space threshold in meters.")
    parser.add_argument("--stacking", action="store_true", help="Enable observation stacking (default matches training).")
    parser.add_argument("--no-stacking", dest="stacking", action="store_false")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env.")
    parser.add_argument("--dt", type=float, default=0.25, help="Time step for env.")


    

    parser.set_defaults(stacking=True)

    args = parser.parse_args()

    if not args.trainer:
        raise ValueError("The --trainer argument must be specified.")

    # model_path = args.model_path
    # if model_path is None and args.trainer.upper() != "BC":
    #     model_path = f"./MODELS/{args.run_id}/{args.run_id}"
    #     print(f"ℹ️ Using default model_path: {model_path}")
    # else:
    #     raise ValueError("model_path must be specified for RL trainers.")

    env = load_eval_env(args.run_id, args.xml, args.stacking, N_STACKING, args.n_envs, args.seed)

    model = load_model(args.trainer, args.run_id, env, args.bc_dir)

    # Evaluate in parallel
    records = evaluate_parallel(env, model, args.episodes, args.safe_dist, args.dt)
    df = pd.DataFrame(records)
    
    per_episode_csv = f"social_metrics_{args.run_id}_{args.trainer.upper()}.csv"
    df.to_csv(per_episode_csv, index=False)

    # Reuse your existing summary utilities (unchanged)
    overall_df = summarize_overall(df)
    per_scen_df = summarize_by_scenario(df)

    overall_csv = f"social_metrics_summary_{args.run_id}_{args.trainer.upper()}.csv"
    per_scen_csv = f"social_metrics_by_scenario_{args.run_id}_{args.trainer.upper()}.csv"
    overall_df.to_csv(overall_csv, index=False)
    per_scen_df.to_csv(per_scen_csv, index=False)

    print("\n=== Overall Social Performance Summary ===")
    for k, v in overall_df.iloc[0].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Per-Scenario Summary (scenario_id) ===")
    print(per_scen_df.to_string(index=False))

        # ---- Histogram of success rates by scenario ----
    plt.figure(figsize=(10, 6))
    plt.bar(per_scen_df["scenario_id"].astype(str), per_scen_df["success_rate"], color="skyblue", edgecolor="black")
    plt.xlabel("Scenario ID")
    plt.ylabel("Success Rate")
    plt.title(f"Success Rates by Scenario - {args.run_id} ({args.trainer.upper()})")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


    print(f"\nSaved per-episode CSV to {per_episode_csv}")
    print(f"Saved overall summary CSV to {overall_csv}")
    print(f"Saved per-scenario summary CSV to {per_scen_csv}")



if __name__ == "__main__":
    main()
