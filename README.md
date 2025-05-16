# 🤖 MuJoCo Human-Aware Mobile Robot RL Navigation

This repository provides an **end-to-end Reinforcement Learning (RL) framework** for training mobile robots to autonomously navigate in indoor environments populated with humans. The framework is built on top of the **MuJoCo physics engine** and uses simulated **laser sensor data** as raw input. The robot learns navigation policies entirely from sensory input to motion output.

## 🧠 Project Objective

Develop a fully autonomous RL agent that enables a mobile robot to:
- Navigate toward target destinations
- Avoid static and dynamic obstacles (including humans)
- Learn socially-aware behaviors through reward shaping or normative modeling
- Operate using only low-dimensional laser-based perception as observations

## ⚙️ Features

- 🧠 **End-to-end RL pipeline** from laser input to velocity commands  
- 🚶 Human-aware navigation using RL  
- 🧭 Goal-reaching with obstacle avoidance  
- 🧩 MuJoCo-based simulation with dynamic environments  
- 🔦 Raw laser scan input as the observation space  
- 📚 Modular training with Stable-Baselines3 algorithms or custom policies  
- 🧪 Evaluation mode for benchmarking trained agents  

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/otr-ebla/MuJoCo_HumanAware_MobileRobot_RLNavigation.git
   cd MuJoCo_HumanAware_MobileRobot_RLNavigation


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
