#!/usr/bin/env python3
import argparse
import os

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from IL_HAMRRLN import il_hamrrln
from tqdm import tqdm

# Default parameters
data_directory = "expert_data"
default_total_steps = 30000 # Total recorded steps 
default_n_envs = 36 # Number of parallel environments that generate data
save_filename = "expert_data.npz"

def make_env(render_mode: str):
    """Return a thunk that creates one env instance in eval mode."""
    def _init():
        return il_hamrrln(render_mode=render_mode, training=False)
    return _init


def collect_bc_data_parallel(
        render_mode: str,
        total_steps: int, # specified via argparse
        n_envs: int,
        save_dir: str,
    ):

    # Ensure data directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create n_envs parallel workers
    env_fns = [make_env(render_mode) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    obs_buffer = []
    act_buffer = []

    steps_collected = 0
    obs = vec_env.reset()  # shape: (n_envs, obs_dim)


    with tqdm(total=total_steps, desc="Collecting IL Data") as pbar:
        while steps_collected < total_steps:
            # 1) Get expert actions from each worker
            
            
            next_obs, _, dones, info = vec_env.step(np.zeros((n_envs, vec_env.action_space.shape[0]), dtype=np.float32))
            expert_actions = np.array(
                [info[i]["expert_action"] for i in range(n_envs)],
                dtype=np.float32,
            )
            

            # expert_actions = np.array(
            #     vec_env.env_method("get_true_robot_velocities"),
            #     dtype=np.float32,
            # )  # shape: (n_envs, action_dim)

            # 2) Record data
            for i in range(n_envs):
                # Always save obs/action before checking if done
                obs_buffer.append(obs[i].copy())
                act_buffer.append(expert_actions[i].copy())

                if dones[i]:
                    # Immediately reset that env to continue generating data
                    reset_result = vec_env.env_method("reset", indices=i)
                    reset_obs, _ = reset_result[0]
                    next_obs[i, :] = reset_obs




            # 3) Step all envs

            obs = next_obs
            steps_collected += n_envs
            pbar.update(n_envs)

    # Stack into two big arrays of shape (total_steps, ...)
    observations = np.vstack(obs_buffer)
    actions = np.vstack(act_buffer)

    # Save only observations & actions, overwrite if exists
    filename = os.path.join(save_dir, "expert_data.npz")
    np.savez(filename, observations=observations, actions=actions)

    print(f"\n✅ Saved {observations.shape[0]} samples to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel expert-data collection for BC (obs + actions only)"
    )
    parser.add_argument(
        "--render-mode", type=str, default="human",
        help="How each env renders (e.g. 'human' or 'rgb_array')"
    )
    parser.add_argument(
        "--total-steps", type=int, default=default_total_steps,
        help="Total transitions to collect across all envs"
    )
    parser.add_argument(
        "--n-envs", type=int, default=default_n_envs,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--save-dir", type=str, default=data_directory,
        help="Directory to write the expert_data.npz file"
    )
    args = parser.parse_args()

    collect_bc_data_parallel(
        render_mode=args.render_mode,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
    )
