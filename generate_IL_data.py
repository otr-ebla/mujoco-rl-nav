import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from IL_HAMRRLN import il_hamrrln
from tqdm import tqdm
import os
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configuration (can still be imported)
DATA_DIR = "expert_data_simple"
EPISODE_LENGTH = 300  
TOTAL_DATA_STEPS = 20000

def collect_expert_data(render_mode="human", save_path=None, total_steps=TOTAL_DATA_STEPS):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Register environment if needed
    env_id = "IL_HAMRRLN-v0"
    try:
        gym.make(env_id)
    except gym.error.UnregisteredEnv:
        gym.envs.registration.register(
            id=env_id,
            entry_point='IL_HAMRRLN:il_hamrrln',
            max_episode_steps=24000,
        )

    # Create environment
    env = DummyVecEnv([lambda: il_hamrrln(render_mode="human", training=False)])

    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_lengths = []

    current_episode = 0
    steps_collected = 0

    with tqdm(total=total_steps, desc="Collecting Expert Data") as pbar:
        while steps_collected < total_steps:
            obs = env.reset()
            done = False
            episode_data = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
            current_obs = obs[0]

            while not done and len(episode_data['observations']) < EPISODE_LENGTH:
                expert_action = env.envs[0].get_true_robot_velocities()
                expert_action_np = np.array(expert_action, dtype=np.float32).reshape(1, -1)

                episode_data['observations'].append(current_obs.copy())
                episode_data['actions'].append(expert_action_np)

                obs, reward, done, info = env.step(expert_action_np)
                current_obs = obs[0]
                episode_data['rewards'].append(reward[0])
                episode_data['dones'].append(done[0])

                pbar.update(1)
                steps_collected += 1

                if done[0] or len(episode_data['observations']) >= EPISODE_LENGTH:
                    all_observations.extend(episode_data['observations'])
                    all_actions.extend(episode_data['actions'])
                    all_rewards.extend(episode_data['rewards'])
                    all_dones.extend(episode_data['dones'])
                    episode_lengths.append(len(episode_data['observations']))
                    current_episode += 1
                    break

    # Save dataset
    filename = save_path or os.path.join(DATA_DIR, "expert_data.npz")
    np.savez(
        filename,
        observations=np.stack(all_observations),
        actions=np.stack(all_actions),
        rewards=np.array(all_rewards),
        dones=np.array(all_dones),
        episode_lengths=np.array(episode_lengths)
    )

    print("\n✅ Data collection complete:")
    print(f"- Episodes: {current_episode}")
    print(f"- Steps: {steps_collected}")
    print(f"- Avg. episode length: {np.mean(episode_lengths):.1f}")
    print(f"- Avg. action magnitude: {np.mean([np.linalg.norm(a) for a in all_actions]):.3f}")
    print(f"- Saved to: {os.path.abspath(filename)}")

# ✅ Guard so it only runs when executed directly
if __name__ == "__main__":
    collect_expert_data(render_mode="human")
