import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from env4walls import RobotNavigationEnv  # Import your environment class

gym.register(
    id="RobotNavigation-v0",
    entry_point="env4walls:RobotNavigationEnv",  # Adjust this to your environment's entry point
    max_episode_steps=1000,  # Set a maximum number of steps per episode
    kwargs={"render_mode": None}  # Pass any necessary arguments to the environment
)

# Create the environment
env_id = "RobotNavigation-v0"
env = gym.make(env_id, render_mode=None)  # No rendering during training

# Create log directory
log_dir = "ppo_robot_nav_logs"
os.makedirs(log_dir, exist_ok=True)

# Create a vectorized environment (optional, for parallel training)
# vec_env = make_vec_env(lambda: gym.make(env_id, render_mode=None), n_envs=4)

# Set up evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
    callback_after_eval=None,
    n_eval_episodes=10,
    verbose=1
)

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
)

# Train the model
total_timesteps = 500000
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    tb_log_name="ppo_robot_nav"
)

# Save the trained model
model.save("ppo_robot_navigation")

# Close the environment
env.close()