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
# Path setup (project root = 2 livelli sopra: .../HumanAwareRLNavigation2)
# Questo file è in: <ROOT>/src/core/lightHAMRRLN.py
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_DIR   = THIS_FILE.parents[1]               # <ROOT>/src
ROOT_DIR  = THIS_FILE.parents[2]               # <ROOT>
ASSETS    = ROOT_DIR / "assets"
DATA_DIR  = ROOT_DIR / "data"
UTILS_DIR = SRC_DIR  / "utils"

# Assicurati che Python veda src/ e la root (per pacchetti data.*)
for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------
# Import locali alla nuova struttura
# ---------------------------------------------------------------------
from core.mobilerobotRL import mobilerobotRL
from core.env_config import *  # costanti del progetto

# Scenari (package: data/scenarios ha __init__.py)
from data.scenarios import (
    scenario1, scenario1_easy, scenario2, scenario3, scenario4, scenario5,
    scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12,
    scenarioTEST1, scenarioTEST2, scenarioTEST3
)

# HSFM/JHSFM (ora sotto src/utils/JHSFM/)
from utils.JHSFM.jhsfm.hsfm import step as hsfm_step
from utils.JHSFM.jhsfm.utils import get_standard_humans_parameters

# Grid decomposition tools (ora sotto src/utils/grid_decomp/)
from utils.grid_decomp.labeled_grid import GridCell_operations

# Keyboard control opzionale
keyboard_active = os.environ.get('KEYBOARD_CONTROL', '0') == '1'
key_pressed = set()
def on_press(key):
    try: key_pressed.add(key.char)
    except AttributeError: pass
def on_release(key):
    try: key_pressed.discard(key.char)
    except AttributeError: pass
keyboard_active = False
if keyboard_active and os.environ.get('DISPLAY', ''):
    try:
        from pynput import keyboard
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
    except Exception as e:
        print(f"⚠️ Warning: Keyboard control requested but failed to initialize listener: {e}")
        keyboard_active = False
else:
    keyboard_active = False


THETA_HIST_LENGTH = 3
N_STACKING = 2
DEBUG = False
DEBUG_DATA = False


class light_hamrrln(mobilerobotRL):
    """
    Optimized Human-Aware Mobile Robot Reinforcement Learning Navigation environment.
    Now includes observation stacking for temporal memory.
    """

    def __init__(
        self,
        num_rays: int = NUM_RAYS,
        model_path: str = None,
        training: bool = True,
        n_humans: int = N_HUMANS,
        render_mode: Optional[str] = None,
        n_stacking: int = N_STACKING,
        enable_stacking: bool = True
    ):
        # Core parameters
        self.num_rays = num_rays
        self.n_humans = n_humans
        self.training = training
        self.render_mode = render_mode
        self.n_stacking = n_stacking
        self.enable_stacking = enable_stacking
        self.lidar_max = MAX_LEN_RAY
        self.use_inverse_lidar = True
        self.dist_floor = float(ROBOT_RADIUS)

        self.waypoints = None
        self.current_wp_index = 0
        self.eval_zero_reward = True   # reward-free eval by default


        # Modello MuJoCo: default sul nuovo path assets/world.xml
        self.model_path = str(ASSETS / "world.xml") if model_path is None else model_path

        # HSFM step compilato
        self.hsfm_step = jax.jit(hsfm_step)

        # Timing parameters
        self.humans_dt = HUMANS_DT
        self.max_episode_time = MAX_EPISODE_TIME

        # Physics parameters
        self.robot_radius = ROBOT_RADIUS

        # Parametri umani (n_humans fissi in training; in eval includiamo robot separatamente)
        self.human_parameters = get_standard_humans_parameters(self.n_humans)

        # Grid / ostacoli
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

        self.current_scenario_id = None
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

        # Import opzionale degli scenari "no humans" (la cartella non è un package)
        self._no_humans = {}
        nh_dir = DATA_DIR / "no_humans_scenariosS"
        if nh_dir.exists():
            try:
                sys.path.insert(0, str(nh_dir.parent))
                for name in [
                    "scenario1_nh","scenario4_nh","scenario5_nh","scenario6_nh",
                    "scenario7_nh","scenario8_nh","scenario9_nh","scenario10_nh",
                    "scenario11_nh","scenario12_nh","scenarioTEST1_nh","scenarioTEST2_nh","scenarioTEST3_nh"
                ]:
                    spec = importlib.util.spec_from_file_location(
                        name, nh_dir / f"{name}.py"
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        self._no_humans[name] = mod
            except Exception as e:
                logging.warning(f"Impossibile caricare scenari no-humans: {e}")

        self.scenario_successes = {i: 0 for i in range(1, len(self.scenario_mapping) + 2)}
        self.scenario_attempts  = {i: 1 for i in range(1, len(self.scenario_mapping) + 2)}  # evita div/0

        # Pre-carica ostacoli per performance
        self._initialize_obstacles()

        # Episode tracking
        self._reset_episode_counters()

        # State variables
        self._initialize_state_variables()

        # Observation stacking
        self._initialize_observation_stacking()

        # Spazi Gym
        self._setup_spaces()

        # Selezione scenario di training/eval
        self.scenarios = [
            scenario4.scenario4,  # abilita se vuoi
            scenario5.scenario5,
            scenario6.scenario6,
            scenario7.scenario7,
            scenario8.scenario8,
            # scenario9.scenario9,  # spesso escluso nel training
            scenario12.scenario12,
            scenarioTEST1.scenarioTEST1,
            scenarioTEST2.scenarioTEST2,
            scenarioTEST3.scenarioTEST3, # perpendicular traffic
        ]

        if len(self.scenarios) < 8:
            msg = "ADDESTRANDO" if self.training else "VALUTANDO"
            print(f"\n\n\n⚠️ STAI {msg} CON POCHI SCENARI! ⚠️\n\n\n")

        if HUMANS_VELOCITY != 1.0:
            print(f"\n\n\n⚠️ STAI USANDO HUMANS_VELOCITY = {HUMANS_VELOCITY} ⚠️\n\n\n")
        else:
            print(f"\n\n\n✅ STAI USANDO HUMANS_VELOCITY = {HUMANS_VELOCITY} ✅\n\n\n")

        self._setup_mujoco()
        self.robot_dt = ROBOT_DT

        # Viewer opzionale
        self.viewer = None
        if self.render_mode == "human":
            self._setup_viewer()

        # Init environment
        self.reset()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _initialize_obstacles(self):
        self.obstacles = None
        self.all_obstacles = False
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=np.float32)

        # Precarica linee dai nuovi path
        self._grid_lines = None
        self._mesh_lines = None
        try:
            with open(UTILS_DIR / "grid_decomp" / "labeled_grid_cleaned.txt", "r") as f:
                self._grid_lines = f.readlines()
        except FileNotFoundError:
            logging.error("Grid file not found (labeled_grid_cleaned.txt)")

        try:
            with open(UTILS_DIR / "grid_decomp" / "mesh_edges.txt", "r") as f:
                self._mesh_lines = f.readlines()
        except FileNotFoundError:
            logging.error("Mesh file not found (mesh_edges.txt)")

        if self.all_obstacles:
            obstacles_data = self._get_all_obstacles()
            if obstacles_data is not None:
                self.obstacles = jnp.stack([obstacles_data] * self.n_humans)

    def set_waypoints(self, waypoints: list[Tuple[float, float]]):
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.eval_zero_reward = True
        self._set_target_position_from_waypoint()

    def _set_target_position_from_waypoint(self):
        if getattr(self, "waypoints", None) is None or len(self.waypoints) == 0:    
            return

        if self.sphere_geom_id >= 0 and self.current_wp_index < len(self.waypoints):
            x, y = self.waypoints[self.current_wp_index]
            self.model.geom_pos[self.sphere_geom_id, :] = [x, y, 2.0]
            self.target_pos = np.array([x, y], dtype=np.float32)

    def _advance_to_next_waypoint(self):
        self.current_wp_index += 1  
        if self.current_wp_index < len(self.waypoints):
            self._set_target_position_from_waypoint()
            self.previous_distance = np.linalg.norm(self.target_pos - self.robot_pos)
            self.last_episode_result = None
            return False
        else:
            self.last_episode_result = "success"
            return True

    def _build_grid_obstacles_per_human(self):
        H = self.n_humans
        M = int(self.max_grid_obs)
        E = int(self.max_edges)
        out = np.full((H, M, E, 2, 2), np.nan, dtype=np.float32)

        for i in range(H):
            x = float(self.humans_state_numpy[i, 0])
            y = float(self.humans_state_numpy[i, 1])

            names = self.grid_cell_op.get_surrounding_obstacle_names_cached(
                x, y, radius=self.grid_radius, grid_size=60
            )
            obs_i = self.grid_cell_op.obstacles_for_names(names)

            if obs_i.size == 0:
                continue

            n = min(obs_i.shape[0], M)
            if obs_i.shape[1] < E:
                pad_edges = np.full((obs_i.shape[0], E - obs_i.shape[1], 2, 2), np.nan, dtype=np.float32)
                obs_i = np.concatenate([obs_i, pad_edges], axis=1)
            elif obs_i.shape[1] > E:
                obs_i = obs_i[:, :E]

            out[i, :n, :, :, :] = obs_i[:n]

        return jnp.array(out, dtype=jnp.float32)  # (H, 10, 4, 2, 2)

    def _reset_episode_counters(self):
        self.current_step = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.human_collision_count = 0
        self.timeout_count = 0
        self.episode_return = 0
        self.previous_distance = 100.0
        self.last_episode_result = None
        self.episode_time_begin = 0
        self.episode_time = 0.0

        self.start_time = time.time()

        self.success_rate = 0.0
        self.collision_rate = 0.0
        self.timeout_rate = 0.0

        self.robot_action_counter = 0
        self.robot_episode_steps = 0
        self.average_step_duration = 0.0
        self.total_step_taken = 0

        self.step_time_measure = 0
        self.robot_action_period = 0.0

        self.progress_reward_scale_initial = 2.0
        self.progress_reward_scale_final = 0.03
        self.progress_reward_term = 0.0
        self.progress_orientation_smoothness_term = 0.0
        self.laser_reward_term = 0.0

        self.last_step_real_time = 0
        self.accumulated_sim_time = 0
        self.real_time_factor = 1.0

        if self.enable_stacking:
            self.polar_stack = deque([np.zeros(2, dtype=np.float32).copy() for _ in range(self.n_stacking)], maxlen=self.n_stacking)

    def _initialize_state_variables(self):
        self.humans_goals = jnp.zeros((self.n_humans, 2, 2), dtype=jnp.float32)
        self.humans_current_goals = jnp.zeros((self.n_humans, 2), dtype=jnp.float32)
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.robot_rot_matrix = np.eye(3, dtype=np.float32)
        self.lidar_readings = np.zeros(self.num_rays, dtype=np.float32)
        self.action = np.zeros(2, dtype=np.float32)
        self.robot_velocity_body = np.zeros(2, dtype=np.float32)
        self.robot_theta = 0.0
        self._theta_hist = deque(maxlen=THETA_HIST_LENGTH)

    def _initialize_observation_stacking(self):
        self.lidar_feat_stack = deque(maxlen=N_STACKING)

    def _setup_spaces(self):
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        K = 24
        feat_per_frame = 3 * K
        total_obs_size = 2 + feat_per_frame * 2

        # bounds per le prime 2 (d, θ)
        goal_low  = np.array([0.0, -np.pi], dtype=np.float32)
        goal_high = np.array([np.finfo(np.float32).max, np.pi], dtype=np.float32)

        obs_low = np.concatenate([
            goal_low,
            np.zeros(feat_per_frame * 2, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            goal_high,
            np.ones(feat_per_frame * 2, dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(total_obs_size,), dtype=np.float32
        )


    def _setup_mujoco(self):
        self.xml_model = self.load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)

        self.mobile_robot_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]
        self.sensor_ids_np = np.asarray(self.lidar_sensor_ids, dtype=np.int32)
        if np.any(self.sensor_ids_np < 0):
            missing = np.where(self.sensor_ids_np < 0)[0].tolist()
            logging.error(f"LIDAR sensors not found for indices: {missing}")

        self.sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if self.sphere_geom_id < 0:
            logging.error("Target geom 'sphere' not found in model.")

        self.human_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            for i in range(self.n_humans)
        ], dtype=np.int32)
        
        self.human_mocap_ids = np.array(
            [self.model.body_mocapid[bid] for bid in self.human_body_ids],
            dtype=np.int32
        )
        if np.any(self.human_mocap_ids < 0):
            missing = np.where(self.human_mocap_ids < 0)[0].tolist()
            logging.error(f"Human mocap bodies not found for indices: {missing}")
    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            try:
                super().reset(seed=seed)
            except Exception:
                np.random.seed(seed)
                random.seed(seed)

        self._update_episode_stats()

        self.episode_count += 1
        self.current_step = 0
        self.episode_time_begin = time.time()

        self.episode_time = 0.0
        self.episode_return = 0.0
        self.last_episode_result = None
        self.relative_angle = 0.0
        self.robot_action_counter = 0
        self.robot_episode_steps = 0
        self.robot_action_period = time.time()
        self.human_update_counter = 0
        self.progress_reward_term = 0.0
        self.laser_reward_term = 0.0
        self.progress_orientation_smoothness_term = 0.0
        self._path_length = 0.0

        robot_id = self.mobile_robot_ID
        self._prev_robot_pos = self.model.body_pos[robot_id, :2].copy()

        mujoco.mj_forward(self.model, self.data)
        super().reset(seed=seed)

        scenario_data = self._load_random_scenario()
        self._set_humans_initial_state(scenario_data)
        self._set_target_position(scenario_data)

        if (not self.training) and (getattr(self, "waypoints", None) is not None) and len(self.waypoints) > 0:
            self.current_wp_index = 0
            self._set_target_position_from_waypoint()

        self.robot_pos = np.array([scenario_data["mob_robot_startposx"], scenario_data["mob_robot_startposy"]])
        self.robot_theta = scenario_data["mob_robot_start_orientation"]
        self.data.qpos[:3] = [self.robot_pos[0], self.robot_pos[1], self.robot_theta]

        self._theta_hist.clear()
        for _ in range(THETA_HIST_LENGTH):
            self._theta_hist.append(self.robot_theta)

        self._L_star = np.linalg.norm(self.target_pos - self.robot_pos)
        self._prev_pos = self.robot_pos.copy()

        self._initialize_humans_tracking()

        info = self._get_info()
        observation = self._get_obs(info)
        mujoco.mj_forward(self.model, self.data)
        return observation, info

    # ------------------------------------------------------------------
    # Osservazioni / Stato
    # ------------------------------------------------------------------
    def _lidar_to_features(self, lidar: np.ndarray, k: int = 24, max_range: float = None, short_r: float = 1.0) -> np.ndarray:
        """Encoder for LiDAR data into compact features (min, mean, short-range min) normalizzati su [0,1]."""
        if max_range is None:
            max_range = float(getattr(self, "lidar_max", 30.0))  # default 30 m

        rays = int(self.num_rays)
        x = np.asarray(lidar[:rays], dtype=np.float32)

        # Clamp sicurezza (se qualche inf è sfuggito): cap a max_range
        x = np.where(np.isfinite(x), x, max_range).astype(np.float32)

        bins = np.array_split(x, k)
        feat_min  = np.array([np.min(b)  for b in bins], dtype=np.float32)
        feat_mean = np.array([np.mean(b) for b in bins], dtype=np.float32)

        short_bins = []
        for b in bins:
            sel = b[b <= short_r]
            if sel.size == 0:
                short_bins.append(np.array([max_range], dtype=np.float32))  # “nessun ostacolo entro short_r”
            else:
                short_bins.append(sel)
        feat_smin = np.array([np.min(b) for b in short_bins], dtype=np.float32)

        if getattr(self, "use_inverse_lidar", False):
            # mappa d → d_floor / max(d, d_floor) ∈ (0,1]; più vicino = più grande
            df = float(getattr(self, "dist_floor", 0.05))
            alpha = float(getattr(self, "inv_alpha", 0.6))
            def inv(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=np.float32)
                # clamp distances from below, elementwise
                arr = np.maximum(arr, df)
                # inverse-with-power, still in [0,1], equals 1 at d=df
                return np.clip((df / arr) ** alpha, 0.0, 1.0).astype(np.float32)
            return np.concatenate([inv(feat_min), inv(feat_mean), inv(feat_smin)]).astype(np.float32)
        else:
            norm = lambda v: np.clip(v / max_range, 0.0, 1.0)
            #out = np.concatenate([norm(feat_min), norm(feat_mean), norm(feat_smin)])
            return np.concatenate([norm(feat_min), norm(feat_mean), norm(feat_smin)]).astype(np.float32)


    def _goal_egocentric(self) -> np.ndarray:
        dxg = float(self.target_pos[0] - self.robot_pos[0])
        dyg = float(self.target_pos[1] - self.robot_pos[1])
        ct, st = np.cos(self.robot_theta), np.sin(self.robot_theta)
        dx =  ct * dxg + st * dyg
        dy = -st * dxg + ct * dyg
        return np.array([dx, dy], dtype=np.float32)

    def _get_state(self) -> np.ndarray:
        current_lidar = np.asarray(self.data.sensordata[self.sensor_ids_np], dtype=np.float32)
        true_current_lidar = current_lidar.copy()

        # Let's apply random noise on current LiDAR readings
        noise_std = 0.04 * self.lidar_max  # 1% del max range
        noise = np.random.normal(0.0, noise_std, size=current_lidar.shape).astype(np.float32)
        current_lidar += noise
        current_lidar = np.clip(current_lidar, 0.0, self.lidar_max).astype(np.float32)



        # Sostituisci i non-finiti (inf/NaN) con la portata massima definita
        if not np.all(np.isfinite(current_lidar)) or current_lidar.size != self.num_rays:
            # mantieni la verifica sulla size, ma non fallire per inf: clamp
            if current_lidar.size != self.num_rays:
                raise RuntimeError("Invalid LiDAR readings; check sensor IDs / model sensors.")
            current_lidar = np.where(np.isfinite(current_lidar), current_lidar, self.lidar_max).astype(np.float32)

        #print("Current LiDAR:", current_lidar)

        delta = self.target_pos[:2] - self.robot_pos[:2]
        distance_to_target = np.linalg.norm(delta)
        target_angle = np.arctan2(delta[1], delta[0])

        robot_orientation = self.robot_theta
        relative_angle = target_angle - robot_orientation
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        self.relative_angle = relative_angle


        return distance_to_target, relative_angle, current_lidar, true_current_lidar


    def _get_obs(self, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        dist, rel_ang, current_lidar, _ = self._get_state()   # <— già calcolati
        # goal egocentrico → POLARE (distanza, angolo)
        goal_vec = np.array([dist, rel_ang], dtype=np.float32)

        feats_t = self._lidar_to_features(current_lidar, k=24, max_range=self.lidar_max, short_r=1.0)
        if len(self.lidar_feat_stack) == 0:
            self.lidar_feat_stack.append(feats_t.copy())
        self.lidar_feat_stack.append(feats_t.copy())
        feats_tm1 = feats_t if len(self.lidar_feat_stack) < 2 else self.lidar_feat_stack[-2]

        observation = np.concatenate([goal_vec, feats_t, feats_tm1]).astype(np.float32)
        return observation


    def _get_info(self) -> Dict[str, Any]:
        if self.sphere_geom_id >= 0:
            sphere_pos = self.model.geom_pos[self.sphere_geom_id]
            distance_to_sphere = float(np.linalg.norm(sphere_pos[:2] - self.robot_pos[:2]))
        else:
            distance_to_sphere = float('inf')

        return {
            "distance_to_sphere": distance_to_sphere,
            "robot_position": np.array([self.robot_pos[0], self.robot_pos[1], float(self.robot_theta)], dtype=np.float32),
            "target_position": self.target_pos.copy(),
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "episode_count": self.episode_count,
            "current_step": self.current_step,
            "scenario_id": self.current_scenario_id,
            "total_observation_size": 146,
        }

    # ------------------------------------------------------------------
    # Step / dinamica
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_obs = self._get_obs()
        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"

        if not self.training:
            current_time = time.time()
            if self.last_step_real_time > 0:
                elapsed_real_time = current_time - self.last_step_real_time
                desired_sim_time = self.robot_dt / self.real_time_factor
                if elapsed_real_time < desired_sim_time:
                    time.sleep(desired_sim_time - elapsed_real_time)
            self.last_step_real_time = time.time()

        self.step_time_measure = time.time()
        self.total_step_taken += 1
        self.robot_action_counter += 1
        self.episode_time = time.time() - self.episode_time_begin
        self.current_step += 1

        if self.training:
            self._apply_robot_action(action, dt=self.robot_dt)

        self._update_humans_simulation(action)

        mujoco.mj_step(self.model, self.data)
        cur_pos = self.robot_pos.copy()
        self._path_length += float(np.linalg.norm(cur_pos - getattr(self, "_prev_pos", cur_pos)))
        self._prev_pos = cur_pos

        self._theta_hist.append(float(self.robot_theta))

        info = self._get_info()
        observation = self._get_obs(info)

        if DEBUG:
            step_reward, terminated, truncated, progress_reward_term, laser_reward_term, theta_smoothness_term = \
                self._calculate_reward_and_termination(info, self.episode_time, observation, prev_obs)
        else:
            step_reward, terminated, truncated = \
                self._calculate_reward_and_termination(info, self.episode_time, observation, prev_obs)

        self.episode_return += step_reward
        info["episode_result"] = self.last_episode_result

        if terminated or truncated:
            info["raw_episode_return"] = float(self.episode_return)
            info["episode_length"] = int(self.current_step)

            if DEBUG:
                print(f"Episode return: {self.episode_return}")
                print(f"Progress reward term: {progress_reward_term}")
                print(f"Laser reward term: {laser_reward_term}")
                print(f"Theta smoothness term: {theta_smoothness_term}")

            if not self.training and self._path_length > 0:
                spl = (self._L_star - 0.5) / self._path_length * 100.0
                print(f"\n\n[DEBUG] Episodio finito ({self.last_episode_result}) - "
                      f"Path length = {self._path_length:.2f} m, "
                      f"L* = {self._L_star-0.5:.2f} m, SPL = {spl:.2f}%\n\n")

        return observation, step_reward, terminated, truncated, info

    def set_real_time_factor(self, factor: float):
        self.real_time_factor = max(0.01, factor)

    # ------------------------------------------------------------------
    # Logica scenari / init
    # ------------------------------------------------------------------
    def _update_episode_stats(self):
        if self.current_step > 0 and self.last_episode_result:
            if self.last_episode_result == "success":
                self.success_count += 1
            elif self.last_episode_result == "collision":
                self.collision_count += 1
            elif self.last_episode_result == "human_collision":
                self.human_collision_count += 1
            elif self.last_episode_result == "timeout":
                self.timeout_count += 1

            if self.current_scenario_id is not None:
                self.scenario_attempts[self.current_scenario_id] += 1
                if self.last_episode_result == "success":
                    self.scenario_successes[self.current_scenario_id] += 1

            if self.episode_count > 1:
                total_episodes = self.episode_count - 1
                self.success_rate  = self.success_count  / total_episodes
                self.collision_rate = self.collision_count / total_episodes
                self.timeout_rate   = self.timeout_count  / total_episodes

                if not self.training:
                    self._log_episode_result(total_episodes)

    def _log_episode_result(self, episode_num: int):
        result_msg = (f"{self.last_episode_result.upper()}: "
                      f"Episode={episode_num} sr={self.success_rate:.2f}, "
                      f"cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, ")
        print(result_msg)

    def _load_random_scenario(self) -> Dict[str, float]:
        if not self.training:
            scenario_func = random.choice(self.scenarios)
            self.current_scenario_id = self.scenario_mapping[scenario_func]
            return scenario_func()

        weights = []
        for func in self.scenarios:
            sid = self.scenario_mapping[func]
            success_rate = self.scenario_successes[sid] / self.scenario_attempts[sid]
            weight = 1.0 - success_rate
            weights.append(max(weight, 0.01))
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()

        scenario_func = random.choices(self.scenarios, weights=weights, k=1)[0]
        self.current_scenario_id = self.scenario_mapping[scenario_func]
        return scenario_func()
    
    def _front_mask_by_index(self, arc_deg: float) -> np.ndarray:
        n = int(self.num_rays)
        half = int(np.ceil((arc_deg / 360.0) * n))   # ampiezza in indici
        mask = np.zeros(n, dtype=bool)
        # Include indici [-half, ..., 0, ..., +half] con wrap-around
        for k in range(-half, half + 1):
            mask[k % n] = True
        return mask

    def _front_arc_stop(self, lidar: np.ndarray, arc_deg: float = 60.0, stop_dist: float = 0.5) -> bool:
        x = np.asarray(lidar[:int(self.num_rays)], dtype=np.float32)
        x = np.where(np.isfinite(x), x, float(self.lidar_max)).astype(np.float32)
        mask = self._front_mask_by_index(arc_deg)
        return float(np.min(x[mask])) < float(stop_dist)


    def _set_humans_initial_state(self, scenario_data: Dict[str, float]):
        OFF_X, OFF_Y, OFF_TH = -1e6, 0.0, 0.0
        human_goals = []

        for i in range(1, self.n_humans + 1):
            x  = float(scenario_data.get(f"human{i}x", OFF_X))
            y  = float(scenario_data.get(f"human{i}y", OFF_Y))
            gx = float(scenario_data.get(f"targethuman{i}x", OFF_X))
            gy = float(scenario_data.get(f"targethuman{i}y", OFF_Y))
            ang = float(scenario_data.get(f"start_orientation_human{i}", OFF_TH))
            if abs(ang) > 2 * np.pi:
                ang = np.deg2rad(ang)

            start_pos  = [x, y]
            target_pos = [gx, gy]
            human_goals.append([start_pos, target_pos])

            #human_id = self.human_body_ids[i - 1]
            # self.model.body_pos[human_id, :2] = start_pos
            # half = ang * 0.5
            # self.model.body_quat[human_id] = [np.cos(half), 0.0, 0.0, np.sin(half)]
            mocap_id = int(self.human_mocap_ids[i - 1])
            # posa iniziale
            self.data.mocap_pos[mocap_id]  = [x, y, 0.0]
            half = ang * 0.5
            self.data.mocap_quat[mocap_id] = [np.cos(half), 0.0, 0.0, np.sin(half)]


        self.humans_goals = jnp.array(human_goals, dtype=jnp.float32)
        self.humans_current_goals = jnp.array([g[0] for g in human_goals], dtype=jnp.float32)

    def _set_target_position(self, scenario_data: Dict[str, float]):
        if self.sphere_geom_id >= 0:
            target_x = scenario_data["target_robot_x"]
            target_y = scenario_data["target_robot_y"]
            self.model.geom_pos[self.sphere_geom_id, :] = [target_x, target_y, 2.0]
        self.target_pos = np.array([scenario_data["target_robot_x"], scenario_data["target_robot_y"]], dtype=np.float32)

    def _initialize_humans_tracking(self):
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=np.float32)
        for i in range(self.n_humans):
            mocap_id = int(self.human_mocap_ids[i])
            px, py, pz = self.data.mocap_pos[mocap_id]
            qw, qx, qy, qz = self.data.mocap_quat[mocap_id]
            theta = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

            self.humans_state_numpy[i, :2] = [px, py]
            self.humans_state_numpy[i, 2:4] = 0.0
            self.humans_state_numpy[i, 4] = theta
            self.humans_state_numpy[i, 5] = 0.0


    # ------------------------------------------------------------------
    # Ricompense / terminazioni
    # ------------------------------------------------------------------
    def _ang_diff(self, a: float, b: float) -> float:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def _safe_div(self, num: float, den: float, eps: float = 1e-8) -> float:
        return float(num) / float(den + eps)

    def _get_max_ang_vel(self) -> float:
        return float(getattr(self, "MAX_ANG_VEL_ROBOT", 1.5))

    def _lasers_reward(self, current_lidar) -> float:
        min_lidar = float(np.min(current_lidar))
        if min_lidar >= LIDAR_THRESHOLD:
            return 0.0
        denom = max(1e-6, (LIDAR_THRESHOLD - float(self.robot_radius)))
        proximity = np.clip((LIDAR_THRESHOLD - min_lidar) / denom, 0.0, 1.0)
        w_lidar = LIDAR_WEIGHT
        return -w_lidar * float(proximity)
    
    def _front_cone_penalty(self, lidar: np.ndarray,
                        arc_deg: float = 90.0,
                        hard: float = None,
                        soft: float = None,
                        weight: float = 0.3) -> float:
        """
        Penalità continua in [-weight, 0]:
        - piena (=-weight) se d_min <= hard
        - zero se d_min >= soft
        - lineare tra hard e soft
        """
        x = np.asarray(lidar[:int(self.num_rays)], dtype=np.float32)
        x = np.where(np.isfinite(x), x, float(self.lidar_max)).astype(np.float32)

        # soglie default coerenti con il robot
        if hard is None: hard = float(self.robot_radius + 0.05)  # stop molto vicino
        if soft is None: soft = float(self.robot_radius + 0.50)  # soglia di “comfort”

        mask = self._front_mask_by_index(arc_deg)
        if not np.any(mask):
            return 0.0

        d_min = float(np.min(x[mask]))

        if d_min <= hard:
            return -float(weight)
        if d_min >= soft:
            return 0.0

        # interpolazione lineare tra hard (penalità piena) e soft (nessuna penalità)
        alpha = (soft - d_min) / max(1e-6, (soft - hard))   # ∈ (0,1)
        return -float(weight) * alpha

    def _orientation_smoothness_penalty(self) -> float:
        if not hasattr(self, "_theta_hist") or len(self._theta_hist) < 2:
            return 0.0
        dt = float(self.robot_dt)
        th = list(self._theta_hist)
        max_w = self._get_max_ang_vel()
        dtheta_t = self._ang_diff(th[-1], th[-2]) / dt
        dtheta_n = dtheta_t / (max_w + 1e-8)
        penalty = 0.0
        lam_d, lam_dd = 0.02, 0.01
        penalty += -lam_d * (dtheta_n ** 2)
        if len(th) >= 3:
            dtheta_tm1 = self._ang_diff(th[-2], th[-3]) / dt
            ddtheta_t = (dtheta_t - dtheta_tm1) / dt
            ddtheta_n = ddtheta_t / (max_w / max(dt, 1e-8))
            penalty += -lam_dd * (ddtheta_n ** 2)
        return float(np.clip(penalty, -0.3, 0.0))

    def _collision_detection(self, current_lidar):
        if np.min(current_lidar) <= ROBOT_RADIUS:
            if DEBUG:
                print(f"Min LIDAR: {np.min(current_lidar):.3f}")
            return True
        return False

    def _calculate_reward_and_termination(self, info: Dict[str, Any], episode_time: float, obs, prev_obs):
        step_reward = 0.0
        terminated, truncated = False, False

        current_target_distance, current_relative_angle, current_lidar, true_current_lidar = self._get_state()

        # --- Collision handling (unchanged) ---
        if self._collision_detection(true_current_lidar):
            self.last_episode_result = "collision"
            terminated = True
            step_reward += -70.0
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = self.episode_time
            info["episode_result"] = "collision"

            # Zero rewards in testing if requested
            if (not self.training) and getattr(self, "eval_zero_reward", False) and (getattr(self, "waypoints", None) is not None):
                step_reward = 0.0

            if DEBUG:
                if not hasattr(self, "progress_reward_term"): self.progress_reward_term = 0.0
                if not hasattr(self, "laser_reward_term"): self.laser_reward_term = 0.0
                if not hasattr(self, "progress_orientation_smoothness_term"): self.progress_orientation_smoothness_term = 0.0
                if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term
            return step_reward, terminated, truncated

        # --- Rewards/penalties (training as before; eval can be zeroed later) ---
        if self.training:
            sr = float(getattr(self, "success_rate", 0.0))
            s0 = float(getattr(self, "progress_reward_scale_initial", 0.04))
            s1 = float(getattr(self, "progress_reward_scale_final",   0.08))
            adaptive_scale = np.clip((1 - sr) * s0 + sr * s1, 0.02, 0.12)
            progress_reward = adaptive_scale * (float(self.previous_distance) - float(current_target_distance))
        else:
            progress_reward = 0.06 * (float(self.previous_distance) - float(current_target_distance))
        progress_reward = float(np.clip(progress_reward, -0.2, 0.2))
        step_reward += progress_reward

        angle_norm = abs(float(current_relative_angle)) / np.pi
        angle_reward = float(np.clip(-0.05 * angle_norm, -0.1, 0.0))
        step_reward += angle_reward

        lidar_reward = float(np.clip(self._lasers_reward(current_lidar), -1.0, 0.0))
        step_reward += lidar_reward

        front_cone_pen = self._front_cone_penalty(
            current_lidar, arc_deg=90.0,
            hard=self.robot_radius + 0.05,
            soft=self.robot_radius + 0.70,
            weight=0.3
        )
        step_reward += front_cone_pen

        theta_smoothness_penalty = float(np.clip(self._orientation_smoothness_penalty(), -0.3, 0.0))
        step_reward += theta_smoothness_penalty

        # --- Success / Waypoint logic ---
        if current_target_distance <= ROBOT_RADIUS + 0.3:
            has_wps = (getattr(self, "waypoints", None) is not None)
            if has_wps and (self.current_wp_index < len(self.waypoints)) and (not self.training):
                # Advance to next waypoint; keep episode alive until final
                finished = self._advance_to_next_waypoint()
                if not finished:
                    # Intermediate waypoint reached
                    info["episode_result"] = "waypoint_reached"
                    info["current_waypoint_index"] = int(self.current_wp_index)
                    # Ensure progress baseline is updated for the new target
                    try:
                        self.previous_distance = float(np.linalg.norm(self.target_pos - self.robot_pos))
                    except Exception:
                        pass
                    # Zero reward in evaluation if requested
                    if getattr(self, "eval_zero_reward", False):
                        step_reward = 0.0

                    if DEBUG:
                        if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                        self.progress_reward_term += progress_reward
                        self.laser_reward_term += lidar_reward
                        self.progress_orientation_smoothness_term += theta_smoothness_penalty
                        self.angle_reward_term += angle_reward
                        return step_reward, False, False, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term
                    return step_reward, False, False

            # Final success (single-goal OR last waypoint)
            self.last_episode_result = "success"
            terminated = True
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = episode_time
            info["episode_result"] = "success"

            # Zero rewards in testing if requested
            if (not self.training) and getattr(self, "eval_zero_reward", False) and (getattr(self, "waypoints", None) is not None):
                step_reward = 0.0
            else:
                step_reward += 200.0

            self.previous_distance = float(current_target_distance)

            if DEBUG:
                if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                self.progress_reward_term += progress_reward
                self.laser_reward_term += lidar_reward
                self.progress_orientation_smoothness_term += theta_smoothness_penalty
                self.angle_reward_term += angle_reward
                return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term
            return step_reward, terminated, truncated

        # --- Timeout (unchanged) ---
        if self.robot_action_counter * self.robot_dt > self.max_episode_time:
            self.last_episode_result = "timeout"
            truncated = True
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = self.episode_time
            info["episode_result"] = "timeout"
            self.previous_distance = float(current_target_distance)

            # Zero rewards in testing if requested
            if (not self.training) and getattr(self, "eval_zero_reward", False) and (getattr(self, "waypoints", None) is not None):
                step_reward = 0.0

            if DEBUG:
                if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                self.progress_reward_term += progress_reward
                self.laser_reward_term += lidar_reward
                self.progress_orientation_smoothness_term += theta_smoothness_penalty
                self.angle_reward_term += angle_reward
                return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_penalty
            return step_reward, terminated, truncated

        # --- Ongoing step (update distance baseline) ---
        self.previous_distance = float(current_target_distance)

        # Zero all step rewards during evaluation with waypoints if requested
        if (not self.training) and getattr(self, "eval_zero_reward", False) and (getattr(self, "waypoints", None) is not None):
            step_reward = 0.0

        if DEBUG:
            if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
            self.progress_reward_term += progress_reward
            self.laser_reward_term += lidar_reward
            self.progress_orientation_smoothness_term += theta_smoothness_penalty
            self.angle_reward_term += angle_reward
            return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term

        return step_reward, terminated, truncated


    # ------------------------------------------------------------------
    # Umani / HSFM
    # ------------------------------------------------------------------
    def _apply_robot_action(self, action: np.ndarray, dt=ROBOT_DT):
        self.action = action
        if not keyboard_active:
            max_lin_vel = MAX_LIN_VEL_ROBOT
            lin_vel = float(action[0]) * max_lin_vel
            ang_vel = float(action[1])

            

            # --- Safety stop: se ostacolo < 0.5 m nei 45° frontali, blocca avanti ---
            current_lidar = np.asarray(self.data.sensordata[self.sensor_ids_np], dtype=np.float32)
            if not np.all(np.isfinite(current_lidar)):
                current_lidar = np.where(np.isfinite(current_lidar), current_lidar, self.lidar_max).astype(np.float32)
            if self._front_arc_stop(current_lidar, arc_deg=80.0, stop_dist=ROBOT_RADIUS + 0.7) and lin_vel > 0.1 and self.previous_distance > 1.8:
                lin_vel = 0.0

            x, y, theta = self.robot_pos[0], self.robot_pos[1], self.robot_theta

            if abs(ang_vel) > 1e-3:
                ratio = lin_vel / ang_vel
                sin_new = np.sin(theta + ang_vel * dt)
                sin_old = np.sin(theta)
                cos_new = np.cos(theta + ang_vel * dt)
                cos_old = np.cos(theta)
                x += ratio * (sin_new - sin_old)
                y += ratio * (-cos_new + cos_old)
                theta += ang_vel * dt
            else:
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                x += lin_vel * cos_theta * dt
                y += lin_vel * sin_theta * dt

            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            self.data.qpos[:3] = np.array([x, y, theta], dtype=np.float32)
            self.robot_pos = np.array([x, y], dtype=np.float32)
            self.robot_theta = theta
        else:
            step_size = 0.01
            rotation_step = 0.01
            yaw = self.data.qpos[2]
            forward_x = np.cos(yaw)
            forward_y = np.sin(yaw)
            if 'u' in key_pressed:
                self.data.qpos[0] += step_size * forward_x
                self.data.qpos[1] += step_size * forward_y
            if 'n' in key_pressed:
                self.data.qpos[0] -= step_size * forward_x
                self.data.qpos[1] -= step_size * forward_y
            if 'g' in key_pressed:
                self.data.qpos[2] += rotation_step
            if 'k' in key_pressed:
                self.data.qpos[2] -= rotation_step

    def _update_humans_simulation(self, action):
        if self.use_grid_obstacles:
            obstacles_agents = self._build_grid_obstacles_per_human()
        else:
            if self.all_obstacles and self.obstacles is not None:
                obstacles_agents = self.obstacles
            else:
                found = self._get_obstacles_from_human_positions(self.humans_state_numpy)
                obstacles_agents = self._get_static_obstacles_formatted(found)

        self._update_human_goals()

        humans_state_jax = jnp.array(self.humans_state_numpy, dtype=jnp.float32)
        n_substeps = int(self.robot_dt / self.humans_dt)

        for _ in range(n_substeps):
            if not self.training:
                self.prev_robot_pos = self.robot_pos.copy()
                self.robot_pos[:] = self.data.qpos[:2]
                robot_theta = self.data.qpos[2]
                self.robot_theta = robot_theta

                vx = (self.robot_pos[0] - self.prev_robot_pos[0]) / self.humans_dt
                vy = (self.robot_pos[1] - self.prev_robot_pos[1]) / self.humans_dt

                rot_matrix = np.array([
                    [np.cos(robot_theta), np.sin(robot_theta)],
                    [-np.sin(robot_theta), np.cos(robot_theta)]
                ])
                self.robot_velocity_body = rot_matrix @ np.array([vx, vy])

                robot_state = jnp.array([
                    self.robot_pos[0], self.robot_pos[1],
                    self.robot_velocity_body[0], self.robot_velocity_body[1],
                    self.robot_theta, 0.0
                ], dtype=jnp.float32)

                humans_state_extended = jnp.concatenate(
                    [humans_state_jax, robot_state[None, :]], axis=0
                )

                robot_goal = jnp.array(self.target_pos[:2], dtype=jnp.float32)
                goals_extended = jnp.concatenate(
                    [self.humans_current_goals, robot_goal[None, :]], axis=0
                )

                robot_params = self.human_parameters[:1]
                params_extended = jnp.concatenate(
                    [self.human_parameters, robot_params], axis=0
                )

                obstacles_extended = jnp.concatenate([obstacles_agents, obstacles_agents[:1]], axis=0)
                humans_state_with_robot = self.hsfm_step(
                    humans_state_extended, goals_extended, params_extended, obstacles_extended, self.humans_dt
                )

                self._apply_robot_action(action, dt=self.humans_dt)
                humans_state_jax = humans_state_with_robot[:-1]
            else:
                humans_state_jax = self.hsfm_step(
                    humans_state_jax, self.humans_current_goals, self.human_parameters, obstacles_agents, self.humans_dt
                )

        if jnp.isnan(humans_state_jax).any():
            logging.warning("NaN values detected in human state")
            return

        self.humans_state_numpy = np.array(humans_state_jax, dtype=np.float32)
        self._update_human_positions_in_mujoco()

    def _update_human_goals(self):
        for i in range(self.n_humans):
            current_pos = self.humans_state_numpy[i, :2]
            current_goal = self.humans_current_goals[i]
            distance_to_goal = np.linalg.norm(current_pos - current_goal)
            if distance_to_goal < 0.1:
                goal_0 = self.humans_goals[i, 0]
                goal_1 = self.humans_goals[i, 1]
                if np.allclose(current_goal, goal_0, atol=1e-6):
                    self.humans_current_goals = self.humans_current_goals.at[i].set(goal_1)
                else:
                    self.humans_current_goals = self.humans_current_goals.at[i].set(goal_0)

    def _update_human_positions_in_mujoco(self):
        for i in range(self.n_humans):
            mocap_id = int(self.human_mocap_ids[i])
            px, py = self.humans_state_numpy[i, 0], self.humans_state_numpy[i, 1]
            th = self.humans_state_numpy[i, 4]
            self.data.mocap_pos[mocap_id]  = [px, py, 0.0]
            self.data.mocap_quat[mocap_id] = [np.cos(th/2.0), 0.0, 0.0, np.sin(th/2.0)]

        # importantissimo: ricomputa kinematics dopo aver mosso i mocap
        mujoco.mj_forward(self.model, self.data)


    # ------------------------------------------------------------------
    # Ostacoli / Grid
    # ------------------------------------------------------------------
    @lru_cache(maxsize=128)
    def _get_obstacles_from_human_positions(self, humans_state_tuple: Tuple) -> List[List[str]]:
        humans_state = np.array(humans_state_tuple).reshape(-1, 6)
        obstacles_per_human = []

        lines = self._grid_lines
        if lines is None:
            grid_file_path = UTILS_DIR / "grid_decomp" / "labeled_grid_cleaned.txt"
            try:
                with open(grid_file_path, "r") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                logging.error("Grid file not found")
                return [[] for _ in range(self.n_humans)]

        for i in range(self.n_humans):
            hx, hy = humans_state[i, 0], humans_state[i, 1]
            cell_x, cell_y = self.grid_cell_op.world_to_grid(hx, hy)

            found_obstacles = []
            for line in lines:
                if line.startswith(f"Cell {cell_x},{cell_y}") or line.startswith(f"Cell {cell_x}, {cell_y}"):
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        obstacle_str = parts[1].strip()
                        if obstacle_str:
                            found_obstacles = obstacle_str.split("|")
                    break

            obstacles_per_human.append(found_obstacles)

        return obstacles_per_human

    @lru_cache(maxsize=1)
    def _get_all_obstacles(self) -> Optional[jnp.ndarray]:
        static_obstacles = []
        lines = self._mesh_lines
        if lines is None:
            mesh_file_path = UTILS_DIR / "grid_decomp" / "mesh_edges.txt"
            try:
                with open(mesh_file_path, "r") as f:
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
                        coord_line = lines[i + j].strip()
                        try:
                            x_str, y_str = coord_line.split(',')
                            vertices.append([float(x_str.strip()), float(y_str.strip())])
                        except ValueError:
                            logging.warning(f"Could not parse coordinate: {coord_line}")
                            continue
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

    # ------------------------------------------------------------------
    # Render/close
    # ------------------------------------------------------------------
    def render(self, mode: str = 'human') -> bool:
        if self.viewer is None:
            self._setup_viewer()
        if mode == 'human' and self.viewer is not None:
            self.viewer.sync()
            return True
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
