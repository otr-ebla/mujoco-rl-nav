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

    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }

    done = False
    current_obs = env.reset()

    with tqdm(total=total_steps, desc="Collecting Expert Data") as pbar:
        while not done and len(episode_data['observations']) < EPISODE_LENGTH:
            prev_obs = current_obs.copy()

            # step the env (IL env ignores the passed action and applies its own)
            dummy = np.zeros((1, 2), dtype=np.float32)
            obs, reward, done, infos = env.step(dummy)

            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            # Try a few common keys for the expert/HSFM command
            expert = None
            for k in ("expert_action", "true_action", "hsfm_action", "expert_cmd"):
                if k in info0:
                    expert = np.array(info0[k], dtype=np.float32).reshape(-1)
                    break
            if expert is None:
                # final fallback: try attributes if you expose them
                try:
                    env0 = env.envs[0]
                    v_lin = float(getattr(env0, "robot_linear_velocity"))
                    v_ang = float(getattr(env0, "robot_angular_velocity"))
                    expert = np.array([v_lin, v_ang], dtype=np.float32)
                except Exception:
                    raise RuntimeError(
                        "Expert action not found in info and no attribute fallback. "
                        "Either emit info['expert_action'] in the env or add a helper method."
                    )

            # record (obs_t, a_t) where a_t is the action applied during this step
            episode_data['observations'].append(prev_obs)
            episode_data['actions'].append(expert)     # store as shape (2,)
            episode_data['rewards'].append(reward[0])
            episode_data['dones'].append(done[0])

            current_obs = obs[0]
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
        actions=np.stack(all_actions),       # <- each is (2,)
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
