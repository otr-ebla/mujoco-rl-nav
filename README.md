# 🤖 MuJoCo Human-Aware Mobile Robot RL Navigation

This repository provides an **end-to-end Reinforcement Learning (RL) framework** for training mobile robots to navigate autonomously in **indoor human-populated environments**, using only **laser sensor data** as input. Built on the **MuJoCo physics engine**, the system allows learning of full behaviors from raw perception to action commands — without any traditional navigation stack.  

The framework is built around the **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** interface and supports multiple state-of-the-art RL algorithms including:
- **TQC** (Truncated Quantile Critics)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)

## 🧠 Project Objective

Develop a fully autonomous RL agent that enables a mobile robot to:
- Navigate to goals in cluttered, dynamic environments
- Avoid both static obstacles and moving humans
- Learn socially-aware and efficient navigation behavior
- Operate using only low-dimensional, raw laser-based perception

## ⚙️ Features

- 🧠 **End-to-end RL pipeline** from sensor input to motion control  
- 🤝 Human-aware navigation with reward shaping or social constraints  
- 📡 Laser range data as the sole observation space  
- 🧩 Realistic simulation via the MuJoCo physics engine  
- ⚙️ **Training support for TQC, SAC, PPO** via Stable-Baselines3  
- 🧪 Evaluation mode for testing trained policies  
- 🧱 Modular environment and training setup for experimentation  

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
