# 🤖 MuJoCo Human-Aware Mobile Robot RL Navigation

<p align="center">
  <img src="assets/image17_trimmed.gif" alt="Environment Rendering Demo" width="650">
</p>

This repository provides an **end-to-end Reinforcement Learning (RL) framework** for training mobile robots to navigate autonomously in **indoor human-populated environments**, using only **laser sensor data** as input.  
Built on the **MuJoCo physics engine**, the system allows learning of full behaviors from raw perception to action commands — without any traditional navigation stack.  

The framework is built around the **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** interface and supports multiple state-of-the-art deep RL algorithms including:
- **TQC** (Truncated Quantile Critics)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)

---

## 🧠 Project Objective

Develop a fully autonomous RL agent that enables a mobile robot to:
- Navigate to goals in cluttered, dynamic environments
- Avoid both static obstacles and moving humans
- Learn socially-aware and efficient navigation behavior
- Operate using only low-dimensional, raw laser-based perception

---

## ⚙️ Features

- 🧠 **End-to-end RL pipeline** from sensor input to motion control  
- 🤝 Human-aware navigation with reward shaping or social constraints  
- 📡 Laser range data as the sole observation space  
- 🧩 Realistic simulation via the MuJoCo physics engine  
- ⚙️ **Training support for TQC, SAC, PPO** via Stable-Baselines3  
- 🧪 Evaluation mode for testing trained policies  
- 🧱 Modular environment and training setup for experimentation  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/otr-ebla/MuJoCo_HumanAware_MobileRobot_RLNavigation.git
   cd MuJoCo_HumanAware_MobileRobot_RLNavigation
   ```

2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv mujoco_env
   source mujoco_env/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Verify MuJoCo installation**
   ```bash
   python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
   ```

---

## 🧩 Project Structure

```
HumanAwareRLNavigation2/
│
├── assets/              # STL meshes, XML world file, policies & callbacks
├── data/                # Scenarios, expert data, social metrics
├── logs/                # Training logs and policy checkpoints
├── models/              # Saved trained models
├── results/             # Plots and aggregated metrics
└── src/
    ├── core/            # Environment and robot physics (lightHAMRRLN, mobilerobotRL)
    ├── RL/              # Training scripts and neural policies
    ├── imitation_learning/  # IL data generation and BC pretraining
    ├── evaluation/      # Evaluation and visualization utilities
    └── utils/           # HSFM, grid decomposition, helper classes
```

---

## 🧠 Using the Environment (`lightHAMRRLN.py`)

`lightHAMRRLN.py` defines the **Gymnasium-compatible environment** that encapsulates:
- Robot motion dynamics
- Human-aware Social Force Model (HSFM)
- Reward shaping and termination logic
- LiDAR observation processing

You can directly instantiate and interact with it for testing:

```bash
source mujoco_env/bin/activate
cd HumanAwareRLNavigation2

python - <<'EOF'
from core.lightHAMRRLN import light_hamrrln

env = light_hamrrln(training=False, render_mode="human")
obs, info = env.reset()
print("Observation shape:", obs.shape)

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
EOF
```

---

## 🚀 Training an RL Agent (`trainHAMR.py`)

The main training and evaluation entrypoint is `src/RL/trainHAMR.py`.  
It supports **TQC**, **SAC**, **PPO**, **TD3**, and **A2C** algorithms.

### ▶️ Basic Training Command

Train a new agent (e.g., TQC) with 8 parallel environments for 10M steps:

```bash
source mujoco_env/bin/activate
cd HumanAwareRLNavigation2/src/RL

python trainHAMR.py --train     --trainer TQC     --num_envs 8     --num_steps 10000000     --run_id TQCrun1    
```

All training logs and checkpoints will be saved in:
```
logs/TENSORBOARD/
logs/policy_checkpoints/
```

You can monitor training progress in TensorBoard:
```bash
tensorboard --logdir logs/TENSORBOARD
```

---

### ⚙️ Other Training Options

- `--trainer [PPO|SAC|TD3|TQC|A2C]` → choose algorithm  
- `--num_envs N` → number of parallel environments  
- `--num_steps N` → total timesteps  
- `--run_id NAME` → unique name for the experiment  
- `--CL` → resume from a previous curriculum learning stage  
- `--bc_path PATH` → load a Behavior Cloning warm-start policy  
- `--render_training` → visualize training (single env only)  
- `--init_from RUNID` → start from weights of another run  
- `--force` → overwrite previous run with same ID  

Example:
```bash
python trainHAMR.py --train --trainer PPO --num_envs 4 --num_steps 5000000 --run_id PPOrun1
```

---

### 🧪 Evaluating a Trained Agent

To test a trained model visually:

```bash
python trainHAMR.py --eval     --trainer TQC     --run_id TQCsuper    
```

This will:
- Load the saved model from `logs/policy_checkpoints/`
- Use the stored normalization stats (`.pkl`)
- Launch MuJoCo viewer for visual playback

---

## 💾 Checkpoints & Normalization Files

- Models are saved as `.zip` in `logs/policy_checkpoints/`
- Normalization stats (`VecNormalize`) saved in `logs/TENSORBOARD/*.pkl`
- You can resume training by reusing the same `--run_id` and adding `--CL`

---

## 🧱 Citation

If you use this repository for your work, please cite:

```bibtex
@software{HumanAwareRLNavigation,
  author = {Vaglio, Alberto},
  title = {Human-Aware RL Navigation in MuJoCo},
  year = {2025},
  url = {https://github.com/otr-ebla/MuJoCo_HumanAware_MobileRobot_RLNavigation}
}
```
---
## Video
[Video of experiments](https://github.com/otr-ebla/MuJoCo_HumanAware_MobileRobot_RLNavigation/blob/main/My%20Movie%201%20-%20SD%20480p.mov)

---

## 📧 Contact

For questions or collaboration:
**alberto.vaglio@student.unisi.it**
