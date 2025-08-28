import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from stable_baselines3 import PPO, SAC, TD3, A2C
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from HAMRRLN import hamrrln
from IL_HAMRRLN import NUM_RAYS, N_STACKING

# Metrics collectors
def compute_metrics(all_infos, success_count, collision_count, timeout_count, episode_lengths, distances_to_human, path_lengths, spl_success_flags, jerks):
    n_episodes = len(all_infos)

    success_rate = success_count / n_episodes
    collision_rate = collision_count / n_episodes
    timeout_rate = timeout_count / n_episodes

    avg_length = np.mean(episode_lengths)
    spl = np.mean([
        (succ * (shortest / actual)) if actual > 0 else 0
        for succ, shortest, actual in zip(spl_success_flags, path_lengths, episode_lengths)
    ])
    min_dist_human = np.min(distances_to_human) if distances_to_human else np.nan
    jerk_mean = np.mean(jerks)

    return success_rate, collision_rate, timeout_rate, avg_length, spl, min_dist_human, jerk_mean


def load_model(trainer, model_path, env):
    if trainer == "PPO": return PPO.load(model_path, env=env)
    if trainer == "SAC": return SAC.load(model_path, env=env)
    if trainer == "TD3": return TD3.load(model_path, env=env)
    if trainer == "A2C": return A2C.load(model_path, env=env)
    if trainer == "TQC": return TQC.load(model_path, env=env)
    raise ValueError(f"Unsupported trainer: {trainer}")


def run_evaluation(model_name, model_path, trainer, env, num_episodes=1000):
    model = load_model(trainer, model_path, env)
    env.training = False

    success_count = 0
    collision_count = 0
    timeout_count = 0
    episode_lengths = []
    spl_success_flags = []
    path_lengths = []
    distances_to_human = []
    jerks = []

    for ep in tqdm(range(num_episodes), desc=f"Evaluating {model_name}"):
        obs = env.reset()
        done = np.array([False])  # default for compatibility
        ep_len = 0
        path = []
        vels = []

        info = {}
        while not done.any():
            action, _ = model.predict(obs, deterministic=True)
            results = env.step(action)
            if len(results) == 4:
                obs, reward, done_flag, info_raw = results
                info = info_raw if isinstance(info_raw, dict) else info_raw[0]
                done = np.array([done_flag])
            else:
                obs, reward, terminated, truncated, infos = results
                info = infos[0] if isinstance(infos, list) else infos
                done = np.logical_or(terminated, truncated)

            ep_len += 1

            robot_pos = info["robot_position"]
            path.append(robot_pos)
            vels.append(np.array(action[0]))  # unbatch action

        result = info["episode_result"]
        if result == "success": success_count += 1
        if result == "collision": collision_count += 1
        if result == "timeout": timeout_count += 1

        shortest_path = np.linalg.norm(info["robot_position"] - info["target_position"])
        actual_path = sum(
            np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            for i in range(len(path)-1)
        )
        spl_success_flags.append(result == "success")
        path_lengths.append(shortest_path)

        min_dist = np.inf
        if env.envs[0].n_humans > 0:
            humans = env.envs[0].humans_state_numpy
            for human in humans:
                dist = np.linalg.norm(human[:2] - info["robot_position"])
                min_dist = min(min_dist, dist)
            distances_to_human.append(min_dist)

        jerk = np.mean(np.linalg.norm(np.diff(vels, axis=0), axis=1)) if len(vels) > 1 else 0
        jerks.append(jerk)

        episode_lengths.append(ep_len)

    return compute_metrics(
        list(range(num_episodes)),
        success_count,
        collision_count,
        timeout_count,
        episode_lengths,
        distances_to_human,
        path_lengths,
        spl_success_flags,
        jerks
    )


def evaluate_all_models(models_dir="MODELS/", trainer="TQC", num_episodes=1000, model_name=None):
    results = []

    if model_name:
        model_path = model_name if model_name.endswith(".zip") else model_name + ".zip"
        model_path = os.path.abspath(model_path)
        model_name = os.path.basename(model_path).replace(".zip", "")
        files = [(model_name, model_path)]
    else:
        files = [
            (f.replace(".zip", ""), os.path.join(models_dir, f))
            for f in os.listdir(models_dir) if f.endswith(".zip")
        ]

    for model_name, model_path in files:
        vec_env = SubprocVecEnv([lambda: hamrrln(
            num_rays=NUM_RAYS,
            training=False,
            enable_stacking=True,
            render_mode=None
        ) for _ in range(4)])

        vecnorm_path = os.path.join("TENSORBOARD", f"{model_name}.pkl")
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        vec_env.training = False

        metrics = run_evaluation(model_name, model_path, trainer, vec_env, num_episodes=num_episodes)

        results.append([
            model_name,
            "ALL",
            *metrics
        ])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="MODELS/", help="Path to directory with trained models")
    parser.add_argument("--trainer", type=str, default="TQC", help="Trainer used: PPO, TQC, SAC, etc.")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Episodes per model")
    parser.add_argument("--model_name", type=str, default=None, help="Name of a single model to test (with or without .zip)")
    args = parser.parse_args()

    all_results = evaluate_all_models(args.models_dir, args.trainer, args.num_episodes, model_name=args.model_name)
    import pandas as pd
    df = pd.DataFrame(all_results, columns=[
        "Model", "Scenario_ID", "Success Rate", "Collision Rate", "Timeout Rate",
        "Avg Episode Length", "SPL", "Min Distance to Human", "Jerk"
    ])
    df.to_csv("evaluation_summary.csv", index=False)
    print("✅ Evaluation complete. Results saved to evaluation_summary.csv")
