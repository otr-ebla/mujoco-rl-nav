import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import argparse
import xml.etree.ElementTree as ET
import os
import torch
from assets.custom_callback import RewardCallback
from HAMRRLN import hamrrln
from nohumans_HAMRRLN import nohumans_hamrrln
from stable_baselines3.common.callbacks import BaseCallback
from assets.custompolicy import TanhActorCriticPolicy
from torch.utils.tensorboard import SummaryWriter

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

