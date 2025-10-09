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
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # JAX solo CPU in eval
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_NUM_THREADS", "1")

# (opzionale ma utile) imposta uno start method sicuro per i worker
import multiprocessing as mp
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

import time
import argparse
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd
import torch, pickle
from math import sqrt


from stable_baselines3 import PPO, SAC, TD3, A2C
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed

#from HAMRRLN import hamrrln
from lightHAMRRLN import light_hamrrln as hamrrln  # versione più leggera senza dipendenze extra
from env_config import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import psutil
import contextlib
import logging

try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:  # fallback minimale
        def __init__(self, total=None, **kwargs): pass
        def update(self, n=1): pass
        def close(self): pass

METRICS_DIR = "./EVAL_METRICS"

# Mappa ID -> nome scenario
SCEN_NAME = {
    14: "parallel traffic",
    15: "intersection",
    16: "perpendicular traffic",
}

def scen_label(x):
    """Ritorna il nome se x è 14/15/16, altrimenti l'originale."""
    try:
        xi = int(x)
    except Exception:
        return str(x)
    return SCEN_NAME.get(xi, str(x))


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
    time_to_goal: float 
    


def load_eval_env(run_id: str, xml: str, stacking: bool, n_stacking: int,                  n_envs: int, seed: int, render: bool) -> VecNormalize:
    base_vec = build_vec_env(n_envs, xml, stacking, n_stacking, seed, render)

    vecnorm_path = os.path.join("./TENSORBOARD", f"{run_id}.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, base_vec)
        print(f"✅ Loaded VecNormalize parameters from {vecnorm_path}")
    else:
        env = VecNormalize(base_vec, norm_obs=False, norm_reward=False)
        print("⚠️ No VecNormalize stats found — proceeding without normalization.")
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

        # Map trainer -> class (TQC potrebbe non essere installato)
        loader_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}
        if TQC is not None:
            loader_map["TQC"] = TQC
        loader = loader_map.get(t)
        if loader is None:
            raise ValueError(f"Unsupported trainer: {trainer}")

        # Device per PyTorch (in eval va benissimo GPU se disponibile)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if t in {"SAC", "TD3", "TQC"}:
            # OFF-POLICY: niente replay buffer gigante in eval
            # - non passiamo env (evita re-init buffer)
            # - sovrascriviamo oggetti salvati per ridurre memoria
            model = loader.load(
                model_path,
                env=None,
                device=device,
                custom_objects={
                    "buffer_size": 1,        # evita allocazioni GB
                    "learning_starts": 0,    # no warmup
                    "replay_buffer": None,   # se presente nel file, annullalo
                },
            )
            # Importante: NON fare model.set_env(env) (ricreerebbe il buffer)
            return model
        else:
            # ON-POLICY (PPO/A2C): nessun replay buffer -> si può caricare con env
            return loader.load(model_path, env=env, device=device)




    # --- BC branch unchanged --- BBBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!
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

# --- stats helper for mean ± 95% CI ---
def _stats_with_ci(series: pd.Series, ci_z: float = 1.96):
    """Return dict with n, mean, std (ddof=1), se, ci95_low/high for a numeric Series."""
    x = pd.to_numeric(series, errors="coerce").dropna().values
    n = int(x.size)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, se=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else np.nan
    se  = float(std / np.sqrt(n)) if n > 1 else np.nan
    ci_low  = float(mean - ci_z * se) if np.isfinite(se) else np.nan
    ci_high = float(mean + ci_z * se) if np.isfinite(se) else np.nan
    return dict(n=n, mean=mean, std=std, se=se, ci95_low=ci_low, ci95_high=ci_high)



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

        try:
            if env.num_envs == 1:
                # VecNormalize → .venv è il VecEnv “sotto”
                base = env.venv
                # DummyVecEnv espone la lista envs
                if hasattr(base, "envs") and len(base.envs) == 1:
                    base.envs[0].render()
        except Exception:
            pass

        steps += 1
        cur_pos = np.array(base_env.robot_pos, dtype=np.float32)
        path_len += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos

        h_raw = base_env.humans_state_numpy
        if h_raw.size > 0:
            h_np = np.asarray(h_raw, dtype=np.float32)
            if h_np.ndim == 1:
                h_np = h_np.reshape(1, -1)
            # se c'è anche il robot come ultima riga, escludilo
            if h_np.shape[0] >= (N_HUMANS + 1):
                humans_xy = h_np[:N_HUMANS, :2]
            else:
                humans_xy = h_np[:, :2]
        else:
            humans_xy = np.empty((0, 2), dtype=np.float32)

        if humans_xy.size > 0:
            step_min = float(np.min(np.linalg.norm(humans_xy - cur_pos[None, :], axis=1)))
            min_dist_human = min(min_dist_human, step_min)
            if step_min >= safe_dist:
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
    dt_env = float(getattr(base_env, "robot_dt", 0.25))

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
        avg_angular_jerk=float('nan'),  # Placeholder, to be computed in parallel evaluation
        time_to_goal=float(steps * dt_env) if last_result == "success" else np.nan
    )



def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    succ = df[df["result"] == "success"]

    # stats we need bounds for
    pl_stats  = _stats_with_ci(succ["path_len"])                 # successes only
    ttg_stats = _stats_with_ci(succ["time_to_goal"])             # successes only
    aj_stats  = _stats_with_ci(df["avg_angular_jerk"])           # all episodes (dropna inside)
    sc_stats = _stats_with_ci(df["space_compliance"])        # all episodes (dropna inside)
    mhd_stats = _stats_with_ci(df["min_dist_human"])        # all episodes (dropna inside)

    s = {
        # counts / rates
        "episodes": len(df),
        "success_rate": (df["result"] == "success").mean(),
        "collision_rate": (df["result"] == "collision").mean(),
        "timeout_rate": (df["result"] == "timeout").mean(),

        # keep your original means for backward-compat
        "avg_path_len_success": succ["path_len"].mean(),
        "mean_time_to_goal_success": succ["time_to_goal"].mean(),
        "mean_avg_angular_jerk": df["avg_angular_jerk"].mean(),
        "mean_SPL": df["spl"].mean(),
        "mean_space_compliance": df["space_compliance"].mean(),

        # NEW: bounded errors (95% CI) + std + n
        "path_len_success_n": pl_stats["n"],
        "path_len_success_mean": pl_stats["mean"],
        "path_len_success_std": pl_stats["std"],
        "path_len_success_ci95_low": pl_stats["ci95_low"],
        "path_len_success_ci95_high": pl_stats["ci95_high"],

        "time_to_goal_success_n": ttg_stats["n"],
        "time_to_goal_success_mean": ttg_stats["mean"],
        "time_to_goal_success_std": ttg_stats["std"],
        "time_to_goal_success_ci95_low": ttg_stats["ci95_low"],
        "time_to_goal_success_ci95_high": ttg_stats["ci95_high"],

        "avg_angular_jerk_n": aj_stats["n"],
        "avg_angular_jerk_mean": aj_stats["mean"],
        "avg_angular_jerk_std": aj_stats["std"],
        "avg_angular_jerk_ci95_low": aj_stats["ci95_low"],
        "avg_angular_jerk_ci95_high": aj_stats["ci95_high"],

        "space_compliance_n": sc_stats["n"],
        "space_compliance_mean": sc_stats["mean"],
        "space_compliance_std": sc_stats["std"],
        "space_compliance_ci95_low": sc_stats["ci95_low"],
        "space_compliance_ci95_high": sc_stats["ci95_high"],

        "min_dist_human_n": mhd_stats["n"],
        "min_dist_human_mean": mhd_stats["mean"],
        "min_dist_human_std": mhd_stats["std"],
        "min_dist_human_ci95_low": mhd_stats["ci95_low"],
        "min_dist_human_ci95_high": mhd_stats["ci95_high"],
    }
    return pd.DataFrame([s])




def summarize_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    def agg_fun(g: pd.DataFrame) -> pd.Series:
        succ = g[g["result"] == "success"]

        pl_stats  = _stats_with_ci(succ["path_len"])
        ttg_stats = _stats_with_ci(succ["time_to_goal"])
        aj_stats  = _stats_with_ci(g["avg_angular_jerk"])
        sc_stats = _stats_with_ci(g["space_compliance"])
        mhd_stats = _stats_with_ci(g["min_dist_human"])

        return pd.Series({
            # counts / rates
            "episodes": len(g),
            "success_rate": (g["result"] == "success").mean(),
            "collision_rate": (g["result"] == "collision").mean(),
            "timeout_rate": (g["result"] == "timeout").mean(),

            # keep existing mean columns
            "avg_path_len_success": succ["path_len"].mean(),
            "mean_time_to_goal_success": succ["time_to_goal"].mean(),
            "mean_avg_angular_jerk": g["avg_angular_jerk"].mean(),
            "mean_SPL": g["spl"].mean(),
            "mean_space_compliance": g["space_compliance"].mean(),

            # NEW: bounded errors (95% CI) + std + n
            "path_len_success_n": pl_stats["n"],
            "path_len_success_mean": pl_stats["mean"],
            "path_len_success_std": pl_stats["std"],
            "path_len_success_ci95_low": pl_stats["ci95_low"],
            "path_len_success_ci95_high": pl_stats["ci95_high"],

            "time_to_goal_success_n": ttg_stats["n"],
            "time_to_goal_success_mean": ttg_stats["mean"],
            "time_to_goal_success_std": ttg_stats["std"],
            "time_to_goal_success_ci95_low": ttg_stats["ci95_low"],
            "time_to_goal_success_ci95_high": ttg_stats["ci95_high"],

            "avg_angular_jerk_n": aj_stats["n"],
            "avg_angular_jerk_mean": aj_stats["mean"],
            "avg_angular_jerk_std": aj_stats["std"],
            "avg_angular_jerk_ci95_low": aj_stats["ci95_low"],
            "avg_angular_jerk_ci95_high": aj_stats["ci95_high"],

            "space_compliance_n": sc_stats["n"],
            "space_compliance_mean": sc_stats["mean"],
            "space_compliance_std": sc_stats["std"],
            "space_compliance_ci95_low": sc_stats["ci95_low"],
            "space_compliance_ci95_high": sc_stats["ci95_high"],

            "min_dist_human_n": mhd_stats["n"],
            "min_dist_human_mean": mhd_stats["mean"],
            "min_dist_human_std": mhd_stats["std"],
            "min_dist_human_ci95_low": mhd_stats["ci95_low"],
            "min_dist_human_ci95_high": mhd_stats["ci95_high"],
        })

    out = df.groupby("scenario_id", dropna=False).apply(agg_fun).reset_index()
    out = out.sort_values("scenario_id", ascending=True)
    return out



def make_env_fn(xml: str, stacking: bool, n_stacking: int, seed: int | None, render_mode: str | None = None):
    def _init():
        env = hamrrln(
            num_rays=NUM_RAYS,
            model_path=xml,
            training=False,
            n_humans=N_HUMANS,
            n_stacking=n_stacking,
            enable_stacking=stacking,
            render_mode=render_mode,
        )
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

def make_env_i(i: int, xml: str, stacking: bool, n_stacking: int, seed: int | None, render_mode: str | None = None):
    """
    Wrapper che imposta l’affinità CPU del worker i
    e poi crea l’ambiente usando la tua make_env_fn.
    """
    def _init():
        # Pinna il processo del worker a un core (best effort)
        try:
            psutil.Process(os.getpid()).cpu_affinity([i % os.cpu_count()])
        except Exception:
            pass
        # Usa la factory esistente, ma con seed differenziato
        return make_env_fn(xml, stacking, n_stacking, (seed + i) if seed is not None else None, render_mode)()
    return _init


def build_vec_env(n_envs: int, xml: str, stacking: bool, n_stacking: int, seed: int,
                  render: bool) -> VecEnv:
    if n_envs <= 1:
        rm = "human" if render else None
        return DummyVecEnv([make_env_fn(xml, stacking, n_stacking, seed, rm)])
    # n_envs > 1 → usa SubprocVecEnv (niente render)
    env_fns = [make_env_i(i, xml, stacking, n_stacking, seed, None) for i in range(n_envs)]
    return SubprocVecEnv(env_fns)



def _extract_omega_from_action(action_vec, info_i, max_ang_vel=None):
    """
    Ritorna l'omega (velocità angolare) *comandata*.
    - Preferisce action dal dict info (se l'env l'ha riscritta/clippata).
    - Altrimenti usa il vettore 'action_vec' passato a env.step.
    - Heuristics:
        dim==2  -> [v, omega]
        dim>=3  -> ultima componente = omega
    - Se max_ang_vel è fornito, scala in rad/s (assumendo azione normalizzata in [-1,1]).
    """
    # 1) prova da info (alcune env salvano l'azione applicata)
    cand = None
    if isinstance(info_i, dict):
        for k in ("action_applied", "last_action", "action"):
            if k in info_i:
                try:
                    arr = np.asarray(info_i[k], dtype=np.float32).ravel()
                    if arr.size >= 2:
                        cand = arr
                        break
                except Exception:
                    pass

    src = cand if cand is not None else np.asarray(action_vec, dtype=np.float32).ravel()
    if src.size < 2:
        # fallback ultra-conservativo: nessuna stima possibile
        return np.nan

    if src.size == 2:
        omega = float(src[1])
    else:
        omega = float(src[-1])  # assume ultima = ω (schema [vx, vy, ω] o simili)

    # scala opzionale (se la policy è normalizzata e l'env usa un cap fisico)
    if max_ang_vel is not None and np.isfinite(max_ang_vel):
        omega = float(np.clip(omega, -1.0, 1.0) * max_ang_vel)

    return omega



def evaluate_parallel(env: VecNormalize, model, episodes_target: int, safe_dist: float, dt: float):
    n_envs = env.num_envs
    obs = env.reset()
    base = env.venv  # SubprocVecEnv or DummyVecEnv
    steps_total = np.zeros(n_envs, dtype=int)
    steps_eval = np.zeros(n_envs, dtype=int)

    def get_attr(name): return base.get_attr(name)
    def env_method(name, indices=None): return base.env_method(name, indices=indices)

    # Obiettivi e baseline
    goals = np.array([np.array(get_attr("target_pos")[i][:2], dtype=np.float32) for i in range(n_envs)], dtype=np.float32)
    last_xy = np.array([p[:2] for p in get_attr("robot_pos")], dtype=np.float32)
    L_star = np.linalg.norm(goals - last_xy, axis=1)

    # dt per ciascun env (se esiste), altrimenti usa quello passato da CLI
    try:
        dt_envs = np.array(get_attr("robot_dt"), dtype=np.float64)
        if not np.all(np.isfinite(dt_envs)):
            dt_envs = np.full(n_envs, float(dt), dtype=np.float64)
    except Exception:
        dt_envs = np.full(n_envs, float(dt), dtype=np.float64)

    # scenario_id
    try:
        infos0 = env_method("_get_info")
        scenario_ids = np.array([int(info.get("scenario_id", -1)) if isinstance(info, dict) else -1 for info in infos0], dtype=int)
    except Exception:
        scenario_ids = np.full(n_envs, -1, dtype=int)

    path_len = np.zeros(n_envs, dtype=np.float64)
    min_dist = np.full(n_envs, np.inf, dtype=np.float64)
    space_ok = np.zeros(n_envs, dtype=int)
    t0       = np.array([time.time()] * n_envs, dtype=np.float64)

    records = []
    completed = 0
    ep_counter = 0

    omega_cmd_history = [[] for _ in range(n_envs)]

    # Prova a leggere un cap fisico per ω (facoltativo, usato per scalare da [-1,1] a rad/s)
    # DOPO: nessuna query ai worker; niente scaling, usiamo azioni raw
    max_ang_vels = np.full(n_envs, np.nan, dtype=np.float64)


    # pbar = tqdm(total=episodes_target, desc=f"Evaluating ({n_envs} envs)", unit="ep", leave=False)
    # prev_disable = logging.root.manager.disable
    # with contextlib.ExitStack() as stack:
    #     devnull = stack.enter_context(open(os.devnull, "w"))
    #     stack.enter_context(contextlib.redirect_stdout(devnull))
    #     logging.disable(logging.CRITICAL)

    while completed < episodes_target:
        actions, _ = model.predict(obs, deterministic=True)

        # ---- registra ω comandata dalla policy per ciascun env ----
        for i in range(n_envs):
            info_i = {}  # info prima dello step non c'è: lascio vuoto, l’estrattore userà 'actions'
            steps_total[i] += 1
            omega_i = _extract_omega_from_action(
                actions[i],
                info_i,
                max_ang_vel=max_ang_vels[i] if i < len(max_ang_vels) else None
            )
            omega_cmd_history[i].append(omega_i)

        # ora esegui lo step
        obs, rewards, dones, infos = env.step(actions)
        try:
            if env.num_envs == 1:
                base = env.venv  # VecNormalize.venv -> DummyVecEnv
                if hasattr(base, "envs") and len(base.envs) == 1:
                    base.envs[0].render()
        except Exception:
            pass


        humans_list  = get_attr("humans_state_numpy")  # può servire come fallback
        

        for i in range(n_envs):
            info_i = infos[i] if isinstance(infos, (list, tuple)) else {}
            # --- posizione e yaw dal dict info (più affidabile) ---
            if isinstance(info_i, dict) and "robot_position" in info_i:
                cur_full = np.array(info_i["robot_position"], dtype=np.float32) 
                cur_xy   = cur_full[:2]
                yaw_val  = float(cur_full[2]) if cur_full.shape[0] >= 3 else 0.0
            else:
                # Fallback: usa get_attr("robot_pos")
                robot_state = np.array(get_attr("robot_pos")[i], dtype=np.float32)
                cur_xy  = robot_state[:2]
                yaw_val = float(robot_state[2]) if robot_state.size >= 3 else 0.0

            # --- path length ---
            path_len[i] += float(np.linalg.norm(cur_xy - last_xy[i]))
            last_xy[i] = cur_xy

            # --- dt per env ---
            dt_i = float(dt_envs[i]) if (dt_envs is not None and np.isfinite(dt_envs[i])) else dt

    

            # --- space compliance (solo umani, mai il robot) ---
            # Preferisci info dal dict se presente
            humans_xy = None
            if isinstance(info_i, dict) and "humans_xy" in info_i:
                arr = np.asarray(info_i["humans_xy"], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    humans_xy = arr[:, :2]

            # Fallback: humans_state_numpy
            if humans_xy is None:
                h_raw = humans_list[i]
                try:
                    h_np = np.asarray(h_raw, dtype=np.float32)
                except Exception:
                    h_np = np.array(h_raw, dtype=np.float32)
                if h_np.ndim == 1:
                    h_np = h_np.reshape(1, -1)
                # Se il formato è [n_humans(+robot), 6], escludi l'ultima riga (robot)
                if h_np.size > 0:
                    # Se il numero di righe è >= N_HUMANS+1 assumiamo ultima = robot
                    if h_np.shape[0] >= (N_HUMANS + 1):
                        humans_xy = h_np[:N_HUMANS, :2]
                    else:
                        humans_xy = h_np[:, :2]

            # Calcolo compliance: se NON ho umani -> metto NaN (non "compliant" di default!)
            if humans_xy is not None and humans_xy.size > 0:
                step_min = float(np.min(np.linalg.norm(humans_xy - (cur_xy[None, :]), axis=1)-HUMANS_RADIUS-ROBOT_RADIUS))
                min_dist[i] = min(min_dist[i], step_min)
                if step_min >= safe_dist:
                    space_ok[i] += 1
                steps_eval[i] += 1  # solo passi con info umani
            else:
                # nessuna informazione sulle posizioni umane → segnala passo non valutabile
                pass


        # --- gestione episodi completati ---
        for i in range(n_envs):
            if dones[i]:
                ep_counter += 1
                result = infos[i].get("episode_result", None)
                if result is None:
                    result = "success" if np.linalg.norm(last_xy[i] - goals[i]) < 0.5 else "timeout"

                elapsed = time.time() - t0[i]
                sim_time = steps_total[i] * float(dt_envs[i])  # tempo simulato
                success_flag = 1.0 if result == "success" else 0.0
                denom = max(path_len[i], L_star[i]-0.5, 1e-6)
                spl = success_flag * (L_star[i]-0.5) / denom

                # space_compliance: usa solo i passi valutabili
                space_comp = (space_ok[i] / steps_eval[i]) if steps_eval[i] > 0 else np.nan

                md = min_dist[i] if np.isfinite(min_dist[i]) else np.nan

                # --- angular jerk (rad/s^3) ---
                # --- angular jerk (da ω comandata) ---
                omegas = np.array(omega_cmd_history[i], dtype=np.float64)
                # pulizia da NaN (può capitare se action dim < 2 in qualche step)
                omegas = omegas[np.isfinite(omegas)]
                if omegas.size >= 3:
                    ang_acc  = np.diff(omegas) / dt_envs[i]      # rad/s^2
                    ang_jerk = np.diff(ang_acc) / dt_envs[i]     # rad/s^3
                    avg_ang_jerk = float(np.mean(np.abs(ang_jerk)))
                else:
                    avg_ang_jerk = np.nan


                records.append({
                    "episode": ep_counter,
                    "scenario_id": int(scenario_ids[i]),
                    "result": str(result),
                    "steps": int(steps_total[i]),
                    "time_s": float(elapsed),
                    "path_len": float(path_len[i]),
                    "min_dist_human": float(md),
                    "start_to_goal": float(L_star[i]),
                    "spl": float(spl),
                    "space_compliance": float(space_comp),
                    "avg_angular_jerk": avg_ang_jerk,
                    "time_to_goal": float(sim_time) if result == "success" else np.nan
                })
                completed += 1
                #pbar.update(1)
                if completed >= episodes_target:
                    break

                # Reset slot i per il prossimo episodio
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
                steps_eval[i] = 0
                steps_total[i] = 0
                space_ok[i] = 0
                t0[i]       = time.time()
                omega_cmd_history[i].clear()

    # logging.disable(prev_disable)
    # pbar.close()
    return records 





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Run/model ID (used for VecNormalize lookup).")
    parser.add_argument("--trainer", type=str, required=True, help="PPO, SAC, TD3, TQC, A2C, or BC")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes.")
    #parser.add_argument("--model_path", type=str, required=False, help="Path to model zip for RL trainers.")
    parser.add_argument("--bc_dir", type=str, default="./bc_policy", help="Dir containing BC artifacts.")
    parser.add_argument("--xml", type=str, default="assets/world.xml", help="MuJoCo XML path.")
    parser.add_argument("--safe_dist", type=float, default=0.5, help="Personal-space threshold in meters.")
    parser.add_argument("--stacking", action="store_true", help="Enable observation stacking (default matches training).")
    parser.add_argument("--no-stacking", dest="stacking", action="store_false")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env.")
    parser.add_argument("--dt", type=float, default=0.25, help="Time step for env.")
    parser.add_argument("--save_only", action="store_true", help="Salva le figure su disco senza mostrarle a schermo")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation (slows down).")


    

    parser.set_defaults(stacking=True)

    args = parser.parse_args()

    if not args.trainer:
        raise ValueError("The --trainer argument must be specified.")
    


    os.makedirs(METRICS_DIR, exist_ok=True)


    env = load_eval_env(args.run_id, args.xml, args.stacking, N_STACKING, args.n_envs, args.seed, args.render)

    model = load_model(args.trainer, args.run_id, env, args.bc_dir)

    # Evaluate in parallel
    records = evaluate_parallel(env, model, args.episodes, args.safe_dist, args.dt)
    df = pd.DataFrame(records)

    # Reuse your existing summary utilities (unchanged)
    overall_df = summarize_overall(df)
    per_scen_df = summarize_by_scenario(df)
    per_scen_spl = (
        df[["scenario_id", "spl"]]
        .dropna(subset=["scenario_id", "spl"])
        .groupby("scenario_id", as_index=False)["spl"]
        .mean()
        .sort_values("scenario_id")
    )

    overall_csv = os.path.join(METRICS_DIR, f"social_metrics_summary_{args.run_id}_{args.trainer.upper()}.csv")
    per_scen_csv = os.path.join(METRICS_DIR, f"social_metrics_by_scenario_{args.run_id}_{args.trainer.upper()}.csv")
    # Per-episode CSV named to match the comparison script's expectations:
    per_episode_csv = os.path.join(METRICS_DIR, f"social_metrics_{args.run_id}_{args.trainer.upper()}.csv")

    overall_df.to_csv(overall_csv, index=False)
    print(f"✅ Overall summary saved to {overall_csv}")

    per_scen_df.to_csv(per_scen_csv, index=False)
    print(f"✅ Per-scenario summary saved to {per_scen_csv}")

    df.to_csv(per_episode_csv, index=False)
    print(f"✅ Per-episode metrics saved to {per_episode_csv}")








    # print("\n=== Overall Social Performance Summary ===")
    # for k, v in overall_df.iloc[0].items():
    #     print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # print("\n=== Per-Scenario Summary (scenario_id) ===")
    # print(per_scen_df.to_string(index=False))

    #     # ---- Histogram of success rates by scenario ----
    # plt.figure(figsize=(10, 6))
    # ids = per_scen_df["scenario_id"].tolist()
    # vals = per_scen_df["success_rate"].tolist()
    # x = np.arange(len(ids))
    # plt.bar(x, vals, edgecolor="black")
    # plt.xlabel("Scenario")
    # plt.ylabel("Success Rate")
    # plt.title(f"Success Rates by Scenario - {args.run_id} ({args.trainer.upper()})")
    # plt.xticks(x, [scen_label(s) for s in ids], rotation=45, ha="right")
    # plt.ylim(0, 1)
    # plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.tight_layout()
    # fig_path = os.path.join(METRICS_DIR, f"success_rates_by_scenario_{args.run_id}_{args.trainer.upper()}.png")
    # plt.savefig(fig_path, dpi=150)
    # if not args.save_only:
    #     plt.show()
    # plt.close()

        # --- BOX PLOT: Time-to-Goal per scenario (solo successi) ---
    # df_success = df[df["result"] == "success"].copy()
    # scen_order = sorted(df_success["scenario_id"].dropna().unique().tolist())

    # if len(scen_order) > 0 and df_success["time_to_goal"].notna().any():
    #     data_by_scen = [df_success.loc[df_success["scenario_id"] == sid, "time_to_goal"].dropna().values
    #                     for sid in scen_order]

    #     plt.figure(figsize=(12, 6))
    #     plt.boxplot(data_by_scen, tick_labels=[scen_label(s) for s in scen_order], showfliers=False)
        

    #     plt.xlabel("Scenario ID")
    #     plt.ylabel("Time to Goal (s)")
    #     plt.title(f"Time-to-Goal by Scenario (Success only) - {args.run_id} ({args.trainer.upper()})")
    #     plt.grid(axis="y", linestyle="--", alpha=0.7)
    #     plt.tight_layout()
    #     fig_path = os.path.join(METRICS_DIR, f"time_to_goal_box_by_scenario_{args.run_id}_{args.trainer.upper()}.png")
    #     plt.savefig(fig_path, dpi=150)
    #     if not args.save_only:
    #         plt.show()
    #     plt.close()




    # # =============== FIGURA COMBINATA 2×3 ===============
    # # Ordine scenari coerente su tutti i pannelli
    
    # scen_order = sorted(df["scenario_id"].dropna().unique().tolist())
    # #x_labels = [str(s) for s in scen_order]
    # x_labels = [scen_label(s) for s in scen_order]

    # # Valori per i bar plot (success rate, SPL medio)
    # sr_map = dict(zip(per_scen_df["scenario_id"].tolist(),
    #                   per_scen_df["success_rate"].tolist()))
    # sr_vals = [float(sr_map.get(s, 0.0)) for s in scen_order]
    # spl_map = dict(zip(per_scen_spl["scenario_id"].tolist(),
    #                    per_scen_spl["spl"].tolist()))
    # spl_vals = [float(spl_map.get(s, 0.0)) for s in scen_order]

    # # (C) Time-to-Goal per scenario (solo successi) → BOX PLOT
    # ttg_df = df.loc[
    #      (df["result"] == "success") & df["time_to_goal"].notna(),
    #      ["scenario_id", "time_to_goal"]
    #  ]
    # ttg_box = [
    #      ttg_df.loc[ttg_df["scenario_id"] == sid, "time_to_goal"].values
    #      for sid in scen_order
    #  ]

    # # (D) Path length (solo successi) → BOX PLOT
    # pl_df = df.loc[
    #     (df["result"] == "success") & df["path_len"].notna(),
    #     ["scenario_id", "path_len"]
    # ]
    # pl_box = [
    #     pl_df.loc[pl_df["scenario_id"] == sid, "path_len"].values
    #     for sid in scen_order
    # ]

    #  # (C) Space compliance medio per scenario (tutti gli episodi) → BAR per scenario
    # sc_df = df.loc[df["space_compliance"].notna(), ["scenario_id", "space_compliance"]]
    # sc_per_scen = (sc_df.groupby("scenario_id")["space_compliance"].mean()
    #                       .reindex(scen_order))
    # sc_vals_by_scen = [float(0.0 if np.isnan(v) else v) for v in sc_per_scen.values]

    # # (F) Jerk angolare medio (tutti gli episodi) → BOX PLOT
    # aj_df = df.loc[df["avg_angular_jerk"].notna(), ["scenario_id", "avg_angular_jerk"]]
    # aj_box = [
    #     aj_df.loc[aj_df["scenario_id"] == sid, "avg_angular_jerk"].values
    #     for sid in scen_order
    # ]
 

    # # Costruisci la figura combinata 2×3
    #         # Costruisci la figura combinata 2×3
    # if len(scen_order) > 0:
    #     fig, axes = plt.subplots(2, 3, figsize=(21, 10), constrained_layout=True)
        

    #     #axes[0,0].bar(x_labels, sr_vals, edgecolor="black")
    #     axes[0,0].bar(x_labels, [v*100.0 for v in sr_vals], edgecolor="black")
    #     axes[0,0].set_title(f"Success Rate per Scenario\n{args.run_id} ({args.trainer.upper()})")
    #     # axes[0,0].set_xlabel("Scenario ID"); axes[0,0].set_ylabel("Success Rate")
    #     # axes[0,0].set_ylim(0.0, 1.0)
    #     axes[0,0].set_xlabel("Scenario ID"); axes[0,0].set_ylabel("Success Rate (%)")
    #     axes[0,0].set_ylim(0.0, 100.0)
    #     axes[0,0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    #     axes[0,0].grid(axis="y", linestyle="--", alpha=0.7)

    #     #axes[0,1].bar(x_labels, spl_vals, edgecolor="black")
    #     axes[0,1].bar(x_labels, [v*100.0 for v in spl_vals], edgecolor="black")
    #     axes[0,1].set_title("Average SPL per Scenario")
    #     axes[0,1].set_xlabel("Scenario ID"); axes[0,1].set_ylabel("Average SPL (%)")
    #     axes[0,1].set_ylim(0.0, 100.0)
    #     axes[0,1].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    #     axes[0,1].grid(axis="y", linestyle="--", alpha=0.7)

    #     # (C) Space compliance
    #     if len(sc_vals_by_scen) > 0:
    #         #axes[0,2].bar(x_labels, sc_vals_by_scen, edgecolor="black")
    #         axes[0,2].bar(x_labels, [v*100.0 for v in sc_vals_by_scen], edgecolor="black")
    #         axes[0,2].set_title("Average Space compliance per scenario")
    #         axes[0,2].set_xlabel("Scenario ID"); axes[0,2].set_ylabel("Compliance (%)")
    #         axes[0,2].set_ylim(0.0, 100.0)
    #         axes[0,2].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    #         axes[0,2].grid(axis="y", linestyle="--", alpha=0.7)
    #     else:
    #         axes[0,2].set_title("Average Space compliance per scenario")
    #         axes[0,2].text(0.5, 0.5, "No data", ha="center", va="center")
    #         axes[0,2].axis("off")

    #     # (D) Path length (successi) — BOX PLOT
    #     if any(len(a) for a in pl_box):
    #         axes[1,0].boxplot(pl_box, tick_labels=x_labels, showfliers=False)
    #         axes[1,0].set_title("Path length (successi) — box plot")
    #         axes[1,0].set_xlabel("Scenario ID"); axes[1,0].set_ylabel("Length (m)")
    #         ymax = float(np.nanmax(pl_df["path_len"])) if not pl_df.empty else 1.0
    #         axes[1,0].set_ylim(0.0, (1.1 * ymax) if ymax > 0 else 1.0)
    #         axes[1,0].grid(axis="y", linestyle="--", alpha=0.7)
    #     else:
    #         axes[1,0].set_title("Path length (successes) — box plot")
    #         axes[1,0].text(0.5, 0.5, "No data", ha="center", va="center")
    #         axes[1,0].axis("off")

    #     # (E) Time-to-Goal (successi) — BOX PLOT
    #     if any(len(a) for a in ttg_box):
    #         axes[1,1].boxplot(ttg_box, tick_labels=x_labels, showfliers=False)
    #         axes[1,1].set_title("Time-to-Goal (successes) — box plot")
    #         axes[1,1].set_xlabel("Scenario ID"); axes[1,1].set_ylabel("Tempo (s)")
    #         ymax = float(np.nanmax(ttg_df["time_to_goal"])) if not ttg_df.empty else 1.0
    #         axes[1,1].set_ylim(0.0, (1.1 * ymax) if ymax > 0 else 1.0)
    #         axes[1,1].grid(axis="y", linestyle="--", alpha=0.7)
    #     else:
    #         axes[1,1].set_title("Time-to-Goal (successes) — box plot")
    #         axes[1,1].text(0.5, 0.5, "No data", ha="center", va="center")
    #         axes[1,1].axis("off")

    #     # (F) Jerk angolare medio — BOX PLOT
    #     if any(len(a) for a in aj_box):
    #         axes[1,2].boxplot(aj_box, tick_labels=x_labels, showfliers=False)
    #         axes[1,2].set_title("Average Angular Jerk — box plot")
    #         axes[1,2].set_xlabel("Scenario ID"); axes[1,2].set_ylabel("rad/s³")
    #         axes[1,2].grid(axis="y", linestyle="--", alpha=0.7)
    #     else:
    #         axes[1,2].set_title("Average Angular Jerk — box plot")
    #         axes[1,2].text(0.5, 0.5, "No data", ha="center", va="center")
    #         axes[1,2].axis("off")

    #     # tilt all x-axis labels
    #     for ax in axes.flatten():
    #         for tick in ax.get_xticklabels():
    #             tick.set_rotation(45)
    #             tick.set_ha("right")

    #     combo_path = os.path.join(METRICS_DIR, f"combined_6panels_{args.run_id}_{args.trainer.upper()}.png")
    #     plt.savefig(combo_path, dpi=150)
    #     print(f"\n\n✅ Saved combined figure to {combo_path} ✅\n\n")
    #     # if not args.save_only:
    #     #     plt.show()
    #     plt.close(fig)


    # # for ax in axes.flatten():
    # #     plt.sca(ax)
    # #     plt.xticks(rotation=45, ha="right")



    # # print(f"\nSaved per-episode CSV to {per_episode_csv}")
    # # print(f"Saved overall summary CSV to {overall_csv}")
    # # print(f"Saved per-scenario summary CSV to {per_scen_csv}")



if __name__ == "__main__":
    main()
