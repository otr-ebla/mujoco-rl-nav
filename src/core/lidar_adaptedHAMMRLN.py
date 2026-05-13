import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
import time
import random
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from pathlib import Path
import importlib

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------
# Path setup (project root = 2 livelli sopra)
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_DIR   = THIS_FILE.parents[1]
ROOT_DIR  = THIS_FILE.parents[2]
ASSETS    = ROOT_DIR / "assets"
DATA_DIR  = ROOT_DIR / "data"
UTILS_DIR = SRC_DIR  / "utils"

for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------
# Imports locali
# ---------------------------------------------------------------------
from src.core.env_config import *  # costanti del progetto
from src.core.mobilerobotRL import mobilerobotRL

from data.scenarios import (
    scenario1, scenario1_easy, scenario2, scenario3, scenario4, scenario5,
    scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12,
    scenarioTEST1, scenarioTEST2, scenarioTEST3
)

from utils.JHSFM.jhsfm.hsfm import step as hsfm_step
from utils.JHSFM.jhsfm.utils import get_standard_humans_parameters
from utils.grid_decomp.labeled_grid import GridCell_operations

# ---------------------------------------------------------------------
# Parametri addizionali per adattamento LiDAR 2D + shaping/"shield"
# ---------------------------------------------------------------------
DEFAULT_NUM_RAYS = 360
OBS_KBINS = 40                 # numero di bin per feature LiDAR
STACK_K = 2                    # frame stacking per memoria breve
R_MAX = 10.0                   # range normalizzato LiDAR
R_SHORT = 1.0                  # sotto-range per feat "short"
R_SAFE = 0.45                  # safety clamp sul lin. vel
R_EMERGENCY = 0.25             # stop se ostacolo molto vicino
SMOOTH_LAMBDA = 0.01           # penalità su delta azioni
CLEAR_LAMBDA = 0.2             # penalità clearance
GOAL_BONUS = 200.0
COLL_PEN = -70.0
STEP_COST = -0.0005

# --- Directional lock params ---
FRONT_SECTOR_DEG = 40.0        # ampiezza del settore frontale (±30°)
LOCK_THRESH      = 0.50        # [m] se front < LOCK_THRESH → blocca v
UNLOCK_THRESH    = 0.65        # [m] per sbloccare v (isteresi > LOCK_THRESH)
UNLOCK_STEPS     = 2           # passi consecutivi sopra UNLOCK_THRESH per sbloccare
#SIDE_BIAS_GAIN   = 1.0         # quanto spingere la rotazione verso il lato più libero
#MIN_TURN_RATE    = 0.35        # ω minimo quando bloccato (frazione di ω_max)
LIDAR_FORWARD_CONVENTION = "center"  # oppure "zero"


# ---------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------
class LidarNav2DEnv(mobilerobotRL):
    """
    Adattamento leggero dell'ambiente per robot mobile con LiDAR 2D in MuJoCo.
    Osservazione compatta con encoder per LiDAR, target egocentrico, velocità/azioni storiche,
    tempo residuo episodio. Include "safety shield" semplice su velocità.
    """

    def __init__(
        self,
        num_rays: int = DEFAULT_NUM_RAYS,
        model_path: Optional[str] = None,
        training: bool = True,
        n_humans: int = N_HUMANS,
        render_mode: Optional[str] = None,
        n_stacking: int = STACK_K,
        enable_stacking: bool = True,
        directional_lock: bool = True,
        safety_shield: bool = True,
    ):
        # Core
        self.num_rays = int(num_rays)
        self.n_humans = int(n_humans)
        self.training = bool(training)
        self.render_mode = render_mode
        self.n_stacking = int(n_stacking)
        self.enable_stacking = bool(enable_stacking)

        self.directional_lock = bool(directional_lock)
        self.safety_shield = bool(safety_shield)
        self._lin_locked = False
        self._front_clear_count = 0

        # Modello MuJoCo
        self.model_path = str(ASSETS / "world.xml") if model_path is None else model_path

        # HSFM
        self.hsfm_step = jax.jit(hsfm_step)

        # Timing / fisica
        self.robot_dt = ROBOT_DT
        self.humans_dt = HUMANS_DT
        self.max_episode_time = MAX_EPISODE_TIME
        self.robot_radius = ROBOT_RADIUS

        # Parametri umani
        self.human_parameters = get_standard_humans_parameters(self.n_humans)

        # Grid ostacoli
        self.grid_cell_op = GridCell_operations(cell_size=4, world_size=320)
        self.max_grid_obs = 10
        self.max_edges = 4
        self.grid_cell_op.load_labeled_grid_from_file(
            str(UTILS_DIR / "grid_decomp" / "labeled_grid_cleaned.txt")
        )
        self.grid_cell_op.load_meshes_index(
            str(UTILS_DIR / "grid_decomp" / "mesh_edges.txt")
        )
        self.grid_radius = 1
        self.use_grid_obstacles = True

        # Scenari (stesso sottoinsieme consigliato per training)
        self.scenario_mapping = {
            scenario1.scenario1: 1,
            scenario2.scenario2: 2,
            scenario3.scenario3: 3,
            scenario4.scenario4: 4,
            scenario5.scenario5: 5,
            scenario6.scenario6: 6,
            scenario7.scenario7: 7,
            scenario8.scenario8: 8,
            scenario9.scenario9: 9,
            scenario10.scenario10: 10,
            scenario11.scenario11: 11,
            scenario12.scenario12: 12,
            scenario1_easy.scenario1_easy: 13,
            scenarioTEST1.scenarioTEST1: 14,
            scenarioTEST2.scenarioTEST2: 15,
            scenarioTEST3.scenarioTEST3: 16,
        }
        self.scenarios = [
            scenario5.scenario5,
            scenario6.scenario6,
            scenario7.scenario7,
            scenario8.scenario8,
            scenario12.scenario12,
            scenarioTEST1.scenarioTEST1,
            scenarioTEST2.scenarioTEST2,
            scenarioTEST3.scenarioTEST3,
        ]

        # Stato episodico/statistiche
        self._reset_counters()
        self._init_state()
        self._init_stacks()

        # Spazi Gym
        self._setup_spaces()

        # MuJoCo
        self._setup_mujoco()
        if self.render_mode == "human":
            self._setup_viewer()

        # Reset iniziale
        self.reset()

    # ------------------------------ Init helpers ------------------------------
    def _reset_counters(self):
        self.episode_count = 0
        self.current_step = 0
        self.episode_return = 0.0
        self.last_episode_result = None
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.success_rate = 0.0
        self.collision_rate = 0.0
        self.timeout_rate = 0.0
        self.scenario_successes = {i: 0 for i in range(1, len(self.scenario_mapping)+2)}
        self.scenario_attempts  = {i: 1 for i in range(1, len(self.scenario_mapping)+2)}

    def _init_state(self):
        self.humans_goals = jnp.zeros((self.n_humans, 2, 2), dtype=jnp.float32)
        self.humans_current_goals = jnp.zeros((self.n_humans, 2), dtype=jnp.float32)
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=np.float32)
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_theta = 0.0
        self.robot_vel_body = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self._theta_hist = deque(maxlen=3)
        self._prev_pos = None
        self._L_star = 0.0

    def _init_stacks(self):
        self.lidar_feat_stack = deque(maxlen=self.n_stacking)

    def _setup_spaces(self):
        # Azioni: v in [0,1] scalato a MAX_LIN_VEL_ROBOT; omega in [-1,1] scalato a MAX_ANG_VEL_ROBOT
        self.action_space = gym.spaces.Box(low=np.array([0.0,-1.0],dtype=np.float32),
                                           high=np.array([1.0, 1.0],dtype=np.float32),
                                           shape=(2,), dtype=np.float32)
        # Osservazione: [goal_vec(2), cosθg,sinθg(2), v,ω (2), vprev,ωprev (2), time_left(1),
        #                lidar_feats_t(3*K), lidar_feats_tm1(3*K)]
        dense_dim = 2 + 2 + 2 + 2 + 1
        feat_per_frame = 3*OBS_KBINS
        total_dim = dense_dim + feat_per_frame*self.n_stacking
        low = np.concatenate([
            np.full(dense_dim, -np.inf, np.float32),
            np.zeros(feat_per_frame*self.n_stacking, np.float32)
        ])
        high = np.concatenate([
            np.full(dense_dim,  np.inf, np.float32),
            np.ones(feat_per_frame*self.n_stacking, np.float32)
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(total_dim,), dtype=np.float32)

    def _setup_mujoco(self):
        # Carica modello
        self.xml_model = self.load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)

        # IDs
        self.mobile_robot_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        self.sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]
        self.sensor_ids_np = np.asarray(self.lidar_sensor_ids, dtype=np.int32)
        if np.any(self.sensor_ids_np < 0):
            missing = np.where(self.sensor_ids_np < 0)[0].tolist()
            logging.error(f"Missing lidar sensors: {missing}")

        self.human_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            for i in range(self.n_humans)
        ], dtype=np.int32)

    # --------------------------- Reset / Step ---------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            try:
                super().reset(seed=seed)
            except Exception:
                np.random.seed(seed)
                random.seed(seed)

        self.episode_count += 1
        self.current_step = 0
        self.episode_return = 0.0
        self.episode_time_begin = time.time()
        self.last_episode_result = None
        self.last_action[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        super().reset(seed=seed)

        # Scenario
        sc_data = self._load_random_scenario()
        self._set_humans_initial_state(sc_data)
        self._set_target_position(sc_data)

        # Pose robot
        self.robot_pos = np.array([sc_data["mob_robot_startposx"], sc_data["mob_robot_startposy"]], dtype=np.float32)
        self.robot_theta = sc_data["mob_robot_start_orientation"]
        self.data.qpos[:3] = [self.robot_pos[0], self.robot_pos[1], self.robot_theta]
        self._theta_hist.clear()
        for _ in range(3):
            self._theta_hist.append(self.robot_theta)
        self._prev_pos = self.robot_pos.copy()
        self._L_star = float(np.linalg.norm(self.target_pos - self.robot_pos))

        # Reset stack
        self.lidar_feat_stack.clear()

        info = self._get_info()
        obs = self._get_obs(info)
        mujoco.mj_forward(self.model, self.data)
        return obs, info

    # --------------------------- Observations ---------------------------
    def _goal_egocentric(self) -> np.ndarray:
        dxg = float(self.target_pos[0] - self.robot_pos[0])
        dyg = float(self.target_pos[1] - self.robot_pos[1])
        ct, st = np.cos(self.robot_theta), np.sin(self.robot_theta)
        dx =  ct * dxg + st * dyg
        dy = -st * dxg + ct * dyg
        return np.array([dx, dy], dtype=np.float32)

    def _bearing_features(self, goal_vec: np.ndarray) -> np.ndarray:
        theta_g = np.arctan2(goal_vec[1], goal_vec[0])
        return np.array([np.cos(theta_g), np.sin(theta_g)], dtype=np.float32)

    def _lidar_to_features(self, lidar: np.ndarray, k: int = OBS_KBINS, max_range: float = R_MAX, short_r: float = R_SHORT) -> np.ndarray:
        bins = np.array_split(lidar[: self.num_rays], k)
        feat_min  = np.array([np.min(b) if b.size>0 else max_range for b in bins], dtype=np.float32)
        feat_mean = np.array([np.mean(b) if b.size>0 else max_range for b in bins], dtype=np.float32)
        feat_smin = []
        for b in bins:
            sel = b[b <= short_r]
            feat_smin.append(np.min(sel) if sel.size>0 else max_range)
        feat_smin = np.array(feat_smin, dtype=np.float32)
        clip01 = lambda x: np.clip(x / max_range, 0.0, 1.0)
        return np.concatenate([clip01(feat_min), clip01(feat_mean), clip01(feat_smin)]).astype(np.float32)

    def _current_lidar(self) -> np.ndarray:
        return np.asarray(self.data.sensordata[self.sensor_ids_np], dtype=np.float32)

    def _time_left_scalar(self) -> float:
        t = time.time() - self.episode_time_begin
        return float(np.clip(1.0 - t / self.max_episode_time, 0.0, 1.0))

    def _get_obs(self, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        lidar = self._current_lidar()
        goal_vec = self._goal_egocentric()
        bearing = self._bearing_features(goal_vec)

        # velocità corpo approx: differenza pos su dt
        cur_pos = self.data.qpos[:2].copy()
        if self._prev_pos is None:
            v_world = np.array([0.0, 0.0], dtype=np.float32)
        else:
            v_world = (cur_pos - self._prev_pos) / max(self.robot_dt, 1e-6)
        self._prev_pos = cur_pos.copy()
        ct, st = np.cos(self.robot_theta), np.sin(self.robot_theta)
        rot = np.array([[ ct,  st],[-st,  ct]], dtype=np.float32)
        self.robot_vel_body = rot @ v_world

        feats_t = self._lidar_to_features(lidar)
        if len(self.lidar_feat_stack) == 0:
            for _ in range(self.n_stacking):
                self.lidar_feat_stack.append(feats_t.copy())
        else:
            self.lidar_feat_stack.append(feats_t.copy())
        lidar_stack = list(self.lidar_feat_stack)
        if len(lidar_stack) < self.n_stacking:
            lidar_stack = [feats_t.copy()] * (self.n_stacking - len(lidar_stack)) + lidar_stack

        dense = np.concatenate([
            goal_vec.astype(np.float32),
            bearing.astype(np.float32),
            np.array([self.robot_vel_body[0] / max(MAX_LIN_VEL_ROBOT,1e-6),
                      (self.data.qvel[2] if self.data.qvel.size>2 else 0.0) / max(getattr(self,'MAX_ANG_VEL_ROBOT',1.5),1e-6)], dtype=np.float32),
            self.last_action.astype(np.float32),
            np.array([self._time_left_scalar()], dtype=np.float32)
        ], dtype=np.float32)

        obs = np.concatenate([dense] + lidar_stack, dtype=np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        if self.sphere_geom_id >= 0:
            sphere_pos = self.model.geom_pos[self.sphere_geom_id]
            distance_to_sphere = float(np.linalg.norm(sphere_pos[:2] - self.robot_pos[:2]))
        else:
            distance_to_sphere = float('inf')
        return {
            "distance_to_goal": distance_to_sphere,
            "robot_position": np.array([self.robot_pos[0], self.robot_pos[1], float(self.robot_theta)], dtype=np.float32),
            "target_position": self.target_pos.copy(),
            "ep_index": self.episode_count,
            "step_idx": self.current_step,
        }

    # --------------------------- Dynamics & HSFM ---------------------------
    def _apply_robot_action(self, action: np.ndarray, dt: float):
        # scala azioni
        v_cmd = float(action[0]) * MAX_LIN_VEL_ROBOT
        w_cmd = float(action[1]) * getattr(self, 'MAX_ANG_VEL_ROBOT', 1.5)

        lidar = self._current_lidar()
        min_r = float(np.nanmin(lidar)) if lidar.size else R_MAX

        # --- Directional lock SOLO sulla direzione di heading ---
        if self.directional_lock:
            # DOPO: usa il settore frontale in frame robot
            front_min = self._front_sector_min(lidar, FRONT_SECTOR_DEG)

            if not self._lin_locked and front_min < LOCK_THRESH:
                self._lin_locked = True
                self._front_clear_count = 0

            if self._lin_locked:
                v_cmd = 0.0
                if front_min >= UNLOCK_THRESH:
                    self._front_clear_count += 1
                    if self._front_clear_count >= UNLOCK_STEPS:
                        self._lin_locked = False
                        self._front_clear_count = 0
                else:
                    self._front_clear_count = 0


        # --- Safety shield SOFT (non tocca mai w_cmd) ---
        # Nota: entra solo se NON sei in lock (quando sei in lock, decide il blocco su heading)
        if self.safety_shield and not self._lin_locked:
            if min_r < R_SAFE:
                scale = np.clip((min_r - R_EMERGENCY) / max(R_SAFE - R_EMERGENCY, 1e-6), 0.0, 1.0)
                v_cmd *= float(scale)
            if min_r < R_EMERGENCY:
                v_cmd = 0.0



        # --- integrazione SE(2) come prima ---
        x, y, th = self.robot_pos[0], self.robot_pos[1], self.robot_theta
        if abs(w_cmd) > 1e-4:
            ratio = v_cmd / w_cmd if abs(w_cmd) > 1e-6 else 0.0
            th_new = th + w_cmd * dt
            x += ratio * (np.sin(th_new) - np.sin(th))
            y += ratio * (-np.cos(th_new) + np.cos(th))
            th = th_new
        else:
            x += v_cmd * np.cos(th) * dt
            y += v_cmd * np.sin(th) * dt
        th = (th + np.pi) % (2*np.pi) - np.pi

        self.data.qpos[:3] = np.array([x, y, th], dtype=np.float32)
        self.robot_pos[:] = [x, y]
        self.robot_theta = th
        self.last_action[:] = np.array([
            v_cmd / max(MAX_LIN_VEL_ROBOT, 1e-6),
            w_cmd / max(getattr(self, 'MAX_ANG_VEL_ROBOT',1.5), 1e-6)
        ], dtype=np.float32)


    def _update_humans_simulation(self):
        # Ostacoli: grid per umano
        obstacles_agents = self._build_grid_obstacles_per_human()
        self._update_human_goals()
        humans_state_jax = jnp.array(self.humans_state_numpy, dtype=jnp.float32)
        n_sub = max(1, int(self.robot_dt / self.humans_dt))
        for _ in range(n_sub):
            humans_state_jax = self.hsfm_step(
                humans_state_jax, self.humans_current_goals, self.human_parameters, obstacles_agents, self.humans_dt
            )
        if jnp.isnan(humans_state_jax).any():
            logging.warning("NaN in human state")
            return
        self.humans_state_numpy = np.array(humans_state_jax, dtype=np.float32)
        self._update_human_positions_in_mujoco()

    # --------------------------- Step / Reward ---------------------------
    def step(self, action: np.ndarray):
        assert action.shape == (2,), f"Expected (2,), got {action.shape}"
        self.current_step += 1

        # dinamica robot + umani
        self._apply_robot_action(action, dt=self.robot_dt)
        self._update_humans_simulation()
        mujoco.mj_step(self.model, self.data)

        info = self._get_info()
        obs = self._get_obs(info)

        # reward + dones
        r, terminated, truncated = self._reward_and_termination()
        r += STEP_COST  # leggero step cost
        self.episode_return += float(r)

        if terminated or truncated:
            info.update({
                'episode_return': float(self.episode_return),
                'episode_len': int(self.current_step),
                'result': self.last_episode_result,
            })
        return obs, float(r), bool(terminated), bool(truncated), info

    def _reward_and_termination(self):
        terminated = False
        truncated = False

        # stato corrente
        goal_delta = self.target_pos - self.robot_pos
        dist = float(np.linalg.norm(goal_delta))
        theta_g = float(np.arctan2(goal_delta[1], goal_delta[0]) - self.robot_theta)
        theta_g = (theta_g + np.pi) % (2*np.pi) - np.pi
        lidar = self._current_lidar()
        min_r = float(np.min(lidar)) if lidar.size else R_MAX

        # progress
        if not hasattr(self, '_prev_dist'):
            self._prev_dist = dist
        sr = float(getattr(self, 'success_rate', 0.0))
        s0, s1 = 0.04, 0.08
        adaptive = np.clip((1 - sr) * s0 + sr * s1, 0.02, 0.12)
        r_prog = adaptive * (self._prev_dist - dist)
        r_prog = float(np.clip(r_prog, -0.2, 0.2))

        # allineamento
        r_angle = -0.05 * abs(theta_g) / np.pi
        r_angle = float(np.clip(r_angle, -0.1, 0.0))

        # clearance / laser penalty
        denom = max(1e-6, (LIDAR_THRESHOLD - float(self.robot_radius)))
        proximity = np.clip((LIDAR_THRESHOLD - min_r) / denom, 0.0, 1.0)
        r_clear = -CLEAR_LAMBDA * float(proximity)
        r_clear = float(np.clip(r_clear, -1.0, 0.0))

        # smoothness su delta azioni
        if not hasattr(self, '_prev_act'):
            self._prev_act = self.last_action.copy()
        da = self.last_action - self._prev_act
        r_smooth = -SMOOTH_LAMBDA * float(da @ da)
        self._prev_act = self.last_action.copy()

        reward = r_prog + r_angle + r_clear + r_smooth

        # success
        if dist <= self.robot_radius + 0.3:
            terminated = True
            self.last_episode_result = 'success'
            reward += GOAL_BONUS
        # collision
        elif min_r <= self.robot_radius:
            terminated = True
            self.last_episode_result = 'collision'
            reward += COLL_PEN
        # timeout
        elif (time.time() - self.episode_time_begin) > self.max_episode_time:
            truncated = True
            self.last_episode_result = 'timeout'

        self._prev_dist = dist
        return reward, terminated, truncated

    # --------------------------- Scenario / Humans ---------------------------
    def _load_random_scenario(self) -> Dict[str, float]:
        # pesi in funzione del successo per curriculum automatico
        weights = []
        for func in self.scenarios:
            sid = self.scenario_mapping[func]
            sr = self.scenario_successes[sid] / self.scenario_attempts[sid]
            weights.append(max(1.0 - sr, 0.01))
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()
        func = random.choices(self.scenarios, weights=weights, k=1)[0]
        self.current_scenario_id = self.scenario_mapping[func]
        return func()

    def _set_humans_initial_state(self, sc: Dict[str, float]):
        human_goals = []
        for i in range(1, self.n_humans + 1):
            x  = float(sc.get(f"human{i}x", -1e6))
            y  = float(sc.get(f"human{i}y", 0.0))
            gx = float(sc.get(f"targethuman{i}x", -1e6))
            gy = float(sc.get(f"targethuman{i}y", 0.0))
            ang = float(sc.get(f"start_orientation_human{i}", 0.0))

            # set stato iniziale per HSFM (x, y, vx, vy, theta, speed)
            self.humans_state_numpy[i-1, 0] = x
            self.humans_state_numpy[i-1, 1] = y
            self.humans_state_numpy[i-1, 2] = 0.0
            self.humans_state_numpy[i-1, 3] = 0.0
            self.humans_state_numpy[i-1, 4] = ang
            self.humans_state_numpy[i-1, 5] = 0.0


            if abs(ang) > 2*np.pi:
                ang = np.deg2rad(ang)

            human_goals.append([[x,y], [gx,gy]])
            hid = self.human_body_ids[i-1]
            self._set_body_pose(hid, np.array([x, y], dtype=np.float32), ang)

            half = ang * 0.5
        self.humans_goals = jnp.array(human_goals, dtype=jnp.float32)
        self.humans_current_goals = jnp.array([g[0] for g in human_goals], dtype=jnp.float32)

    def _set_target_position(self, sc: Dict[str, float]):
        if self.sphere_geom_id >= 0:
            tx, ty = sc["target_robot_x"], sc["target_robot_y"]
            self.model.geom_pos[self.sphere_geom_id, :] = [tx, ty, 2.0]
        self.target_pos = np.array([sc["target_robot_x"], sc["target_robot_y"]], dtype=np.float32)

    def _update_human_goals(self):
        for i in range(self.n_humans):
            pos = self.humans_state_numpy[i, :2]
            cur = self.humans_current_goals[i]
            if np.linalg.norm(pos - cur) < 0.1:
                g0, g1 = self.humans_goals[i, 0], self.humans_goals[i, 1]
                self.humans_current_goals = self.humans_current_goals.at[i].set(g1 if np.allclose(cur, g0) else g0)

    def _update_human_positions_in_mujoco(self):
        for i in range(self.n_humans):
            hid = self.human_body_ids[i]
            pos = self.humans_state_numpy[i, :2]
            th  = self.humans_state_numpy[i, 4]
            self._set_body_pose(hid, pos, th)


    def _front_sector_min(self, lidar: np.ndarray, sector_deg: float = FRONT_SECTOR_DEG) -> float:
        """Ritorna il min range nel settore frontale (centrato sulla direzione del robot)."""
        if lidar.size == 0:
            return R_MAX
        w = max(1, int((sector_deg / 360.0) * self.num_rays / 2))
        mid = self.num_rays // 2
        lo  = max(0, mid - w)
        hi  = min(self.num_rays, mid + w)
        return float(np.min(lidar[lo:hi]))
    
    def _heading_sector_min(self, lidar: np.ndarray, sector_deg: float) -> float:
        """Min del range in un settore centrato sull'heading del robot."""
        if lidar.size == 0:
            return R_MAX

        # centro del settore (indice del raggio 'in avanti' rispetto al muso)
        if LIDAR_FORWARD_CONVENTION == "center":
            base = self.num_rays // 2
        else:  # "zero"
            base = 0

        # heading corrente del robot (in radianti)
        theta = float(self.robot_theta)
        # passi di indice corrispondenti a theta (CCW positivo)
        offset = int(round((theta / (2*np.pi)) * self.num_rays))
        center = (base + offset) % self.num_rays

        w = max(1, int((sector_deg / 360.0) * self.num_rays / 2))
        # finestra [center-w, center+w] con wrap
        idxs = (np.arange(center - w, center + w + 1) % self.num_rays).astype(int)
        return float(np.nanmin(lidar[idxs]))

    def _set_body_pose(self, body_id: int, pos_xy: np.ndarray, yaw: float):
        # prova free joint
        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num > 0:
            j_id = self.model.jnt_type[jnt_adr]
            # 0=free, 1=ball, 2=slide, 3=hinge  (free ha 7 dof in qpos)
            if j_id == mujoco.mjtJoint.mjJNT_FREE:
                qadr = self.model.jnt_qposadr[jnt_adr]
                # qpos free = [x y z qw qx qy qz]
                self.data.qpos[qadr + 0] = pos_xy[0]
                self.data.qpos[qadr + 1] = pos_xy[1]
                self.data.qpos[qadr + 2] = 0.0
                half = yaw * 0.5
                qw = np.cos(half); qz = np.sin(half)
                self.data.qpos[qadr + 3:qadr + 7] = [qw, 0.0, 0.0, qz]
                return

        # prova mocap (se il body ha una geom/weld ad un mocap body)
        # -> qui serve conoscere l'indice del mocap; se non lo usi, salta

        # fallback (non dinamico): modifica model e fai forward
        self.model.body_pos[body_id, :2] = [pos_xy[0], pos_xy[1]]
        half = yaw * 0.5
        self.model.body_quat[body_id] = [np.cos(half), 0.0, 0.0, np.sin(half)]























    # --------------------------- Ostacoli / Grid ---------------------------
    @lru_cache(maxsize=1)
    def _get_all_obstacles(self) -> Optional[jnp.ndarray]:
        static_obstacles = []
        try:
            with open(UTILS_DIR / "grid_decomp" / "mesh_edges.txt", "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            logging.error("mesh_edges.txt not found")
            return None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith(":"):
                vertices = []
                for j in range(1, 5):
                    if i + j < len(lines):
                        xs, ys = lines[i + j].strip().split(',')
                        vertices.append([float(xs.strip()), float(ys.strip())])
                if len(vertices) == 4:
                    edges = [
                        [vertices[0], vertices[1]],
                        [vertices[1], vertices[2]],
                        [vertices[2], vertices[3]],
                        [vertices[3], vertices[0]],
                    ]
                    static_obstacles.append(edges)
                i += 4
            i += 1
        return jnp.array(static_obstacles) if static_obstacles else None

    def _build_grid_obstacles_per_human(self):
        H, M, E = self.n_humans, int(self.max_grid_obs), int(self.max_edges)
        out = np.full((H, M, E, 2, 2), np.nan, dtype=np.float32)
        try:
            with open(UTILS_DIR / "grid_decomp" / "labeled_grid_cleaned.txt", "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            logging.error("Grid file not found")
            return jnp.array(out)
        for i in range(H):
            x = float(self.humans_state_numpy[i, 0])
            y = float(self.humans_state_numpy[i, 1])
            names = self.grid_cell_op.get_surrounding_obstacle_names_cached(x, y, radius=self.grid_radius, grid_size=60)
            obs_i = self.grid_cell_op.obstacles_for_names(names)
            if obs_i.size == 0:
                continue
            n = min(obs_i.shape[0], M)
            if obs_i.shape[1] < E:
                pad = np.full((obs_i.shape[0], E - obs_i.shape[1], 2, 2), np.nan, dtype=np.float32)
                obs_i = np.concatenate([obs_i, pad], axis=1)
            elif obs_i.shape[1] > E:
                obs_i = obs_i[:, :E]
            out[i, :n, :, :, :] = obs_i[:n]
        return jnp.array(out, dtype=jnp.float32)

    # --------------------------- Viewer / Close ---------------------------
    def render(self, mode: str = 'human') -> bool:
        if not hasattr(self, 'viewer') or self.viewer is None:
            self._setup_viewer()
        if mode == 'human' and self.viewer is not None:
            self.viewer.sync()
            return True
        return False

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
