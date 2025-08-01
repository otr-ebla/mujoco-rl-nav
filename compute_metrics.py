#!/usr/bin/env python3
"""
Use evaluate_policy with a custom callback to collect per-episode metrics in parallel envs without rendering,
wrapped with a Monitor to correctly catch episode ends, and display a progress bar instead of prints.

Usage:
  # Evaluate and log raw per-episode data:
  python compute_metrics.py eval \
      --policy-path my_policy.zip \
      --xml-path path/to/env_model.xml \
      --trainer TQC \
      --n-episodes 200 \
      --num-envs 8 \
      --csv-out results.csv \
      --norm-path vecnormalize.pkl \
      --sc-threshold 0.7

  # Compute summary metrics from CSV:
  python compute_metrics.py summary \
      --csv-in results.csv \
      --summary-out summary.csv
"""
import os
import sys
import contextlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO, SAC, TD3, A2C
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from HAMRRLN import hamrrln


def make_eval_env(args):
    """Create a monitored vectorized env to allow evaluate_policy to track episodes"""
    def _init():
        return hamrrln(
            num_rays=args.num_rays,
            model_path=args.xml_path,
            training=False,
            render_mode=None,
            n_humans=args.n_humans,
            n_stacking=args.n_stacking,
            enable_stacking=not args.no_stacking
        )
    env = DummyVecEnv([_init] * args.num_envs)
    if args.norm_path and os.path.exists(args.norm_path):
        env = VecNormalize.load(args.norm_path, env)
    # wrap with VecMonitor so evaluate_policy sees 'dones'
    env = VecMonitor(env)
    env.training = False
    env.norm_reward = False
    return env


def load_model(args, env):
    cls_map = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'TQC': TQC, 'A2C': A2C}
    cls = cls_map.get(args.trainer.upper())
    if cls is None:
        raise ValueError(f"Unsupported trainer: {args.trainer}")
    return cls.load(args.policy_path, env=env)


class MetricsCallback(BaseCallback):
    """
    Collects MDH, SC, episode length, and jerk per episode during evaluation,
    updating a tqdm progress bar on each episode completion.
    """
    def __init__(self, threshold, env, pbar, verbose=0):
        super().__init__(verbose)
        self.sc_thr = threshold
        self.env = env
        self.pbar = pbar
        self.episode_data = []
        self._reset_buffers()

    def _reset_buffers(self):
        self.min_dists = []
        self.jerks = []
        self.step_count = 0
        self.prev_vel = None
        self.prev_acc = None

    def _on_step(self) -> bool:
        idx = self.locals.get('env_idx', 0)
        env_inst = self.env.envs[idx]
        # record minimum distance to humans
        dists = np.linalg.norm(env_inst.humans_state_numpy[:, :2] - env_inst.robot_pos, axis=1)
        self.min_dists.append(dists.min())
        # jerk calculation
        vel = env_inst.robot_velocity_body
        dt = env_inst.robot_dt
        if self.step_count == 0:
            self.prev_vel = vel
            self.prev_acc = np.zeros_like(vel)
        acc = (vel - self.prev_vel) / dt
        jerk = (acc - self.prev_acc) / dt
        self.jerks.append(np.linalg.norm(jerk))
        self.prev_vel = vel
        self.prev_acc = acc
        self.step_count += 1

        dones = self.locals.get('dones', [False] * self.env.num_envs)
        # monitor episode finishes across parallel envs
        for env_idx, done in enumerate(dones):
            if done:
                info = self.locals['infos'][env_idx]
                result = info.get('episode_result', 'unknown')
                ep_time = info.get('episode_time_length', self.step_count * dt)
                mdh = float(np.min(self.min_dists))
                avg_jerk = float(np.mean(self.jerks))
                sc = float(np.mean(np.array(self.min_dists) >= self.sc_thr))
                self.episode_data.append({
                    'result': result,
                    'steps': self.step_count,
                    'episode_time': ep_time,
                    'mdh': mdh,
                    'avg_jerk': avg_jerk,
                    'social_compliance': sc,
                })
                self.pbar.update(1)
                self._reset_buffers()
        return True


def evaluate_and_log(args):
    env = make_eval_env(args)
    model = load_model(args, env)
    pbar = tqdm(total=args.n_episodes, desc='Evaluating', file=sys.stderr)
    metrics_cb = MetricsCallback(threshold=args.sc_threshold, env=env, pbar=pbar)

    def eval_callback(locals_, globals_):
        metrics_cb.locals = locals_
        metrics_cb.globals = globals_
        return metrics_cb._on_step()

    # suppress all prints from environment (stdout & stderr)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            evaluate_policy(
                model,
                env,
                n_eval_episodes=args.n_episodes,
                deterministic=True,
                render=False,
                callback=eval_callback,
                return_episode_rewards=False
            )
    pbar.close()

    # Save metrics CSV
    df = pd.DataFrame(metrics_cb.episode_data)
    df.index += 1
    df.insert(0, 'episode', df.index)
    df.to_csv(args.csv_out, index=False)


def compute_summary(args):
    df = pd.read_csv(args.csv_in)
    summary = {
        'success_rate': (df['result'] == 'success').mean(),
        'collision_rate': (df['result'] == 'collision').mean(),
        'timeout_rate': (df['result'] == 'timeout').mean(),
        'mean_mdh': df['mdh'].mean(),
        'mean_social_compliance': df['social_compliance'].mean(),
        'mean_episode_length': df['episode_time'].mean(),
        'mean_jerk': df['avg_jerk'].mean(),
    }
    print("=== SUMMARY METRICS ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    if args.summary_out:
        pd.Series(summary).to_csv(args.summary_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Eval mode
    p = subparsers.add_parser('eval')
    p.add_argument('--policy-path', required=True, help='Path to trained SB3 policy .zip')
    p.add_argument('--xml-path', required=True, help='Path to MuJoCo XML file for the environment')
    p.add_argument('--trainer', required=True)
    p.add_argument('--n-episodes', type=int, default=200)
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--csv-out', required=True)
    p.add_argument('--num-rays', type=int, default=108)
    p.add_argument('--n-humans', type=int, default=5)
    p.add_argument('--n-stacking', type=int, default=10)
    p.add_argument('--no-stacking', action='store_true')
    p.add_argument('--norm-path', default=None, help='VecNormalize stats .pkl')
    p.add_argument('--sc-threshold', type=float, default=0.7)

    # Summary mode
    ps = subparsers.add_parser('summary')
    ps.add_argument('--csv-in', required=True)
    ps.add_argument('--summary-out', default=None)

    args = parser.parse_args()
    if args.mode == 'eval':
        evaluate_and_log(args)
    else:
        compute_summary(args)
