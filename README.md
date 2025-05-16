# 🤖 MuJoCo Human-Aware Mobile Robot RL Navigation
This repository contains a Reinforcement Learning (RL) framework for training mobile robots to navigate in human-populated indoor environments using the MuJoCo physics engine. The robot uses simulated **laser sensor data** to perceive its surroundings and learns to navigate safely and efficiently while being aware of nearby humans.

## 🧠 Project Objective

Develop a reinforcement learning agent that enables a mobile robot to:
- Navigate toward target destinations
- Avoid static and dynamic obstacles (including humans)
- Learn human-aware behaviors using reward shaping or social norms
- Operate with limited laser-based perception

## ⚙️ Features

- 🚶 Human-aware navigation using RL
- 🧭 Goal-reaching with obstacle avoidance
- 🧩 MuJoCo-based simulation with dynamic environments
- 🔦 Laser scan input as observation space
- 📚 Modular training with algorithms from Stable-Baselines3 or custom implementations
- 🧪 Evaluation mode for trained agents

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/otr-ebla/MuJoCo_HumanAware_MobileRobot_RLNavigation.git
   cd MuJoCo_HumanAware_MobileRobot_RLNavigation

2. **Create and activate a Python virtual environment**
    ```bash
    python3 -m venv mujoco_env
    source mujoco_env/bin/activate     # On Windows: venv\Scripts\activate

2. **Create and activate a Python virtual environment**
    ```bash
    pip install -r requirements.txt
