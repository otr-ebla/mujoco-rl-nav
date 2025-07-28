from stable_baselines3 import PPO
import gymnasium as gym

# Load the trained model
model = PPO.load("ppo_robot_navigation")

# Create environment with rendering
env = gym.make("RobotNavigation-v0", render_mode="human")

# Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()