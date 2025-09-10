import numpy as np
import gymnasium as gym
import mujoco
# NOTE: non importo mujoco.viewer qui per evitare side effects in headless;
# lo useremo solo dentro _setup_viewer() se mai servirà.
from mobilerobotRL import mobilerobotRL
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import xml.etree.ElementTree as ET
import time
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any

import logging
from collections import deque

# Scenari
# Scenari
from scenarios import (
    scenario1, scenario1_easy, scenario2, scenario3, scenario4, scenario5,
    scenario6, scenario7, scenario8, scenario9, scenario10, scenario11,
    scenario12,
    scenarioTEST1, scenarioTEST2, scenarioTEST3,   # NEW: match RL env
)


import jax.numpy as jnp
from JHSFM.jhsfm.hsfm import step
from JHSFM.jhsfm.utils import get_standard_humans_parameters
from grid_decomp.labeled_grid import GridCell_operations
from assets.collisondetector import CollisionDetector




# =========================
# Costanti di ambiente
# =========================
from env_config import NUM_RAYS, N_STACKING, MAX_LIN_VEL_ROBOT, DISTANCE_SUCCESS_THRESHOLD, MAX_EPISODE_TIME, ROBOT_RADIUS, LIDAR_THRESHOLD, ROBOT_DT, HUMANS_DT, N_HUMANS, PROGRESS_REWARD_SCALE


COLLISION_THRESHOLD = 0.4
REPELLENT_FORCE = 0.35
REPELLENT_WALL_FORCE = 1.0

# Rendering disabilitato di default in training
rendering_disable = True

# Dummy stato tastiera (rimane sempre vuoto: niente controllo manuale in headless)
keyboard_active = False
key_pressed = set()
TOGGLE_KEY  = 'm'
FWD_KEY     = 'u'
BACK_KEY    = 'n'
LEFT_KEY    = 'k'
RIGHT_KEY   = 'g'


class il_hamrrln(mobilerobotRL):
    """
    Env per Imitation Learning: il robot segue dinamica tipo "human" via HSFM.
    Nessun input da tastiera in headless.
    """

    def __init__(self,
                 num_rays: int = NUM_RAYS,
                 model_path: str = "assets/world.xml",
                 training: bool = True,
                 n_humans: int = 5,
                 render_mode: Optional[str] = None,
                 n_stacking: int = N_STACKING):

        # Core params
        self.num_rays = num_rays
        self.n_humans = n_humans
        self.training = training
        self.render_mode = None
        self.model_path = model_path
        self.n_stacking = n_stacking
        self._prev_robot_pos = np.zeros(2, dtype=np.float32)
        self.initial_robot_distance = 0.0

        # Manual control (disattivato in headless)
        self.manual_control = False
        self._last_toggle_time = 0.0
        self.il_buffer = []

        # Timing
        self.humans_dt = HUMANS_DT
        self.max_episode_time = MAX_EPISODE_TIME

        # Physics
        self.robot_radius = ROBOT_RADIUS

        # Parametri HSFM
        self.human_parameters = get_standard_humans_parameters(self.n_humans + 1)
        self.human_parameters = self.human_parameters.at[-1, 2].set(MAX_LIN_VEL_ROBOT)
        self.human_parameters = self.human_parameters.at[-1, 6].set(REPELLENT_FORCE)
        self.human_parameters = self.human_parameters.at[-1, 10].set(REPELLENT_WALL_FORCE)

        self.grid_cell_op = GridCell_operations(cell_size=4, world_size=320)

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
            scenarioTEST1.scenarioTEST1: 14,               # NEW
            scenarioTEST2.scenarioTEST2: 15,               # NEW
            scenarioTEST3.scenarioTEST3: 16,               # NEW
        }


        # Pre-carica ostacoli
        self._initialize_obstacles()

        # Episode tracking
        self._reset_episode_counters()

        # Variabili di stato
        self._initialize_state_variables()

        # Stacking osservazioni
        self._initialize_observation_stacking()

        # Spazi Gym
        self._setup_spaces()

        # Sottoinsieme scenari per IL
        self.scenarios = [
            ##########scenario1.scenario1, # Incrocio caos
            ###############scenario4.scenario4, # Corridoio - PARALLEL TRAFFIC
            
            scenario5.scenario5, # Scenario con robot che si muove tra 3 tavoli con umani
            scenario6.scenario6, # Scenario con robot attravers DUE PORTE
            scenario7.scenario7, # Scenario con robot che gira tra le colonne
            scenario8.scenario8, # Scenario in fondo, con robot che attraversa la porta con 3 umani nel mezzo

            scenario9.scenario9, # PERPEDICULAR TRAFFIC

            scenario12.scenario12, # Scenario EASY, stanza all'inizio a destra
            scenarioTEST1.scenarioTEST1, # TEST - PARALLEL TRAFFIC
            scenarioTEST2.scenarioTEST2, # TEST - Incrocio caos
            #scenarioTEST3.scenarioTEST3, # TEST - PERPEDICULAR TRAFFIC

            # scenario 5 può essere usato come test
        ]

        # MuJoCo
        self._setup_mujoco()

        human_names = [f"human{i+1}" for i in range(self.n_humans)]
        self.collision_detector = CollisionDetector(
            self.model, self.data,
            robot_body_name="agent_body",
            human_body_names=human_names
        )
        self.robot_dt = ROBOT_DT

        # Viewer (non creato se rendering_disable=True)
        self.viewer = None

        # Reset iniziale
        self.reset()

    # ----------------------------
    # Utilità (senza tastiera)
    # ----------------------------
    def _keyboard_action(self) -> np.ndarray:
        """Ritorna azione da 'stato tasti' (qui sempre vuoto in headless)."""
        lin, ang = 0.0, 0.0
        if FWD_KEY in key_pressed: lin += 1.0
        if BACK_KEY in key_pressed: lin -= 1.0
        if LEFT_KEY in key_pressed: ang += 1.0
        if RIGHT_KEY in key_pressed: ang -= 1.0
        return np.array([np.clip(lin, 0.0, 1.0), np.clip(ang, -1.0, 1.0)], dtype=np.float32)

    def _apply_manual_action(self, action: np.ndarray, dt=ROBOT_DT):
        """Applica azione diretta al robot (non usata in headless)."""
        lin_vel = float(action[0]) * MAX_LIN_VEL_ROBOT
        ang_vel = float(action[1])

        x, y, theta = self.data.qpos[:3]
        if abs(ang_vel) > 0.001:
            R = lin_vel / ang_vel if ang_vel != 0 else 0.0
            x += R * (np.sin(theta + ang_vel * dt) - np.sin(theta))
            y -= R * (np.cos(theta) - np.cos(theta + ang_vel * dt))
            theta += ang_vel * dt
        else:
            x += lin_vel * np.cos(theta) * dt
            y += lin_vel * np.sin(theta) * dt

        self.data.qpos[:3] = [x, y, theta]

    # ----------------------------
    # Init helpers
    # ----------------------------
    def _initialize_obstacles(self):
        self.obstacles = None
        self.all_obstacles = True
        self.humans_state_numpy = np.zeros((self.n_humans + 1, 6), dtype=np.float32)
        if self.all_obstacles:
            obstacles_data = self._get_all_obstacles()
            if obstacles_data is not None:
                self.obstacles = jnp.stack([obstacles_data] * (self.n_humans + 1))

    def _reset_episode_counters(self):
        self.current_step = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.episode_return = 0
        self.human_collision_count = 0
        self.previous_distance = 100.0
        self.last_episode_result = None
        self.episode_time_begin = 0
        self.episode_time = 0.0
        self.human_update_counter = 0
        self.start_time = time.time()

        self.success_rate = 0.0
        self.collision_rate = 0.0
        self.timeout_rate = 0.0
        self.human_collision_rate = 0.0
        self.robot_action_counter = 0

        self.robot_episode_steps = 0
        self.average_step_duration = 0.0
        self.total_step_taken = 0
        self.step_time_measure = 0
        self.robot_action_period = 0.0
        self.polar_stack = deque([np.zeros(2, dtype=np.float32).copy() for _ in range(self.n_stacking)], maxlen=self.n_stacking)

    def _initialize_state_variables(self):
        self.humans_goals = jnp.zeros((self.n_humans + 1, 2, 2), dtype=jnp.float32)
        self.humans_current_goals = jnp.zeros((self.n_humans + 1, 2), dtype=jnp.float32)
        self.robot_pos = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.robot_rot_matrix = np.eye(3, dtype=np.float32)

    def _initialize_observation_stacking(self):
        self.lidar_stack = deque(maxlen=self.n_stacking)
        empty_lidar = np.zeros(self.num_rays, dtype=np.float32)
        for _ in range(self.n_stacking):
            self.lidar_stack.append(empty_lidar.copy())

    def _setup_spaces(self):
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        stacked_lidar_size = self.num_rays * self.n_stacking
        stacked_polar_size = 2 * self.n_stacking

        obs_low = np.concatenate([
            np.zeros(stacked_lidar_size, dtype=np.float32),        # lidar
            np.zeros(self.n_stacking, dtype=np.float32),           # distances
            np.full(self.n_stacking, -np.pi, dtype=np.float32),    # angles
        ])
        obs_high = np.concatenate([
            np.full(stacked_lidar_size, 200.0, dtype=np.float32),  # lidar
            np.full(self.n_stacking, 200.0, dtype=np.float32),     # distances
            np.full(self.n_stacking,  np.pi, dtype=np.float32),    # angles
        ])
        total_obs_size = stacked_polar_size + stacked_lidar_size

        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_obs_size,),
            dtype=np.float32
        )

    def _setup_mujoco(self):
        self.xml_model = self.load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)
        self.mobile_robot_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        self.human_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            for i in range(self.n_humans)
        ], dtype=np.int32)
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]

    # ----------------------------
    # Reset / Step
    # ----------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.training and self.episode_time > 0:
            _ = time.time() - self.start_time
        self._update_episode_stats()

        self.episode_count += 1
        self.current_step = 0
        self.previous_distance = 70.0
        self.episode_time_begin = time.time()
        self.episode_time = 0.0
        self.last_episode_result = None
        self.relative_angle = 0.0
        self.robot_action_counter = 0
        self.robot_episode_steps = 0
        self.robot_action_period = time.time()
        self.human_update_counter = 0

        robot_id = self.mobile_robot_ID
        self._prev_robot_pos = self.model.body_pos[robot_id, :2].copy()

        mujoco.mj_forward(self.model, self.data)
        super().reset(seed=seed)

        scenario_data = self._load_random_scenario()
        self._set_humans_initial_state(scenario_data)
        self._initialize_humans_tracking()
        self._reset_observation_stack()

        info = self._get_info()
        observation = self._get_obs(info)
        mujoco.mj_forward(self.model, self.data)
        return observation, info

    def _reset_observation_stack(self):
        self.lidar_stack.clear()
        self.polar_stack.clear()
        initial_robot_distance, initial_relative_angle, initial_lidar = self._get_state()
        for _ in range(self.n_stacking):
            self.lidar_stack.append(initial_lidar.copy())
        empty_polar = np.array([initial_robot_distance, initial_relative_angle], dtype=np.float32)
        for _ in range(self.n_stacking):
            self.polar_stack.append(empty_polar.copy())

    def _update_lidar_stack(self, new_lidar_reading: np.ndarray):
        self.lidar_stack.append(new_lidar_reading.copy())

    def _get_stacked_lidar_obs(self) -> np.ndarray:
        stacked = np.stack(list(self.lidar_stack), axis=0)
        return stacked.flatten()

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

            total_episodes = self.episode_count
            if total_episodes > 1:
                self.success_rate = self.success_count / total_episodes
                self.collision_rate = self.collision_count / total_episodes
                self.human_collision_rate = self.human_collision_count / total_episodes
                self.timeout_rate = self.timeout_count / total_episodes

    def _load_random_scenario(self) -> Dict[str, float]:
        scenario_func = random.choice(self.scenarios)
        self.current_scenario_id = self.scenario_mapping.get(scenario_func, 0)
        return scenario_func()

    def _set_humans_initial_state(self, scenario_data: Dict[str, float]):
        human_positions = []
        human_goals = []

        for i in range(self.n_humans):
            human_pos = (scenario_data[f"human{i+1}x"], scenario_data[f"human{i+1}y"])
            human_goal = (scenario_data[f"targethuman{i+1}x"], scenario_data[f"targethuman{i+1}y"])
            orientation = scenario_data[f"start_orientation_human{i+1}"]
            human_positions.append(human_pos)
            human_goals.append(human_goal)
            human_id = self.human_body_ids[i]
            self.model.body_pos[human_id, :2] = [human_pos[0], human_pos[1]]
            self.model.body_quat[human_id] = [np.cos(orientation / 2), 0.0, 0.0, np.sin(orientation / 2)]

        robot_pos = (scenario_data["mob_robot_startposx"], scenario_data["mob_robot_startposy"])
        robot_goal = (scenario_data["target_robot_x"], scenario_data["target_robot_y"])
        robot_orientation = scenario_data["mob_robot_start_orientation"]
        human_positions.append(robot_pos)
        human_goals.append(robot_goal)

        self.target_pos[:2] = robot_goal

        target_sphere_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if target_sphere_id >= 0:
            self.model.geom_pos[target_sphere_id, :2] = [robot_goal[0], robot_goal[1]]
        else:
            logging.warning("Target sphere geom not found in MuJoCo model.")

        robot_id = self.mobile_robot_ID
        self.model.body_pos[robot_id, :2] = [robot_pos[0], robot_pos[1]]
        self.model.body_quat[robot_id] = [np.cos(robot_orientation / 2), 0.0, 0.0, np.sin(robot_orientation / 2)]

        goals_array = []
        current_goals = []
        for pos, goal in zip(human_positions, human_goals):
            goals_array.append([goal, pos])
            current_goals.append(goal)

        self.humans_goals = jnp.array(goals_array, dtype=jnp.float32)
        self.humans_current_goals = jnp.array(current_goals, dtype=jnp.float32)

    def _initialize_humans_tracking(self):
        self.humans_state_numpy = np.zeros((self.n_humans + 1, 6), dtype=np.float32)

        for i in range(self.n_humans):
            human_id = self.human_body_ids[i]
            self.humans_state_numpy[i, :2] = self.model.body_pos[human_id, :2]
            self.humans_state_numpy[i, 2:4] = 0.0
            quat = self.model.body_quat[human_id]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            theta = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self.humans_state_numpy[i, 4] = theta
            self.humans_state_numpy[i, 5] = 0.0

        robot_id = self.mobile_robot_ID
        self.humans_state_numpy[self.n_humans, :2] = self.model.body_pos[robot_id, :2]
        self.humans_state_numpy[self.n_humans, 2:4] = 0.0
        quat = self.model.body_quat[robot_id]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        theta = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.humans_state_numpy[self.n_humans, 4] = theta
        self.humans_state_numpy[self.n_humans, 5] = 0.0

    def _get_state(self) -> np.ndarray:
        self.robot_pos[:2] = self.humans_state_numpy[-1, :2]

        current_lidar = np.zeros(self.num_rays, dtype=np.float32)
        for i, sensor_id in enumerate(self.lidar_sensor_ids):
            if sensor_id >= 0:
                current_lidar[i] = round(self.data.sensordata[sensor_id], 2)
            else:
                raise ValueError(f"Invalid sensor ID for lidar_{i}: {sensor_id}")

        delta = self.target_pos[:2] - self.robot_pos[:2]
        distance_to_target = np.linalg.norm(delta)
        target_angle = np.arctan2(delta[1], delta[0])

        robot_orientation = self.humans_state_numpy[-1, 4]
        relative_angle = target_angle - robot_orientation
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        self.relative_angle = relative_angle
        return distance_to_target, relative_angle, current_lidar

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        previous_obs = self._get_obs()
        current_step_duration = time.time() - self.step_time_measure
        self.total_step_taken += 1
        info = self._get_info()

        self.average_step_duration = (self.average_step_duration * (self.total_step_taken - 1) +
                                      (time.time() - self.step_time_measure)) / self.total_step_taken
        if self.average_step_duration > 100:
            self.average_step_duration = 0.008

        self.step_time_measure = time.time()
        self.episode_time = time.time() - self.episode_time_begin
        self.robot_action_counter += 1

        # Nessun toggle manuale in headless (key_pressed è sempre vuoto)

        if self.manual_control:
            action = self._keyboard_action()
            self._apply_manual_action(action, dt=self.robot_dt)
            obs = self._get_obs()
            self.il_buffer.append((obs.copy(), action.copy()))
        else:
            humansstate = self.humans_state_numpy.copy()
            self._update_humans_simulation()
            next_humans_state = self.humans_state_numpy.copy()

        actual_angle = next_humans_state[-1, 4]
        prev_angle = humansstate[-1, 4]
        diff_angle = actual_angle - prev_angle
        wrapped_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
        ang_vel = wrapped_angle / self.robot_dt
        lin_vel = np.linalg.norm(next_humans_state[-1, 0:2] - humansstate[-1, 0:2]) / self.robot_dt

        mujoco.mj_forward(self.model, self.data)
        self.current_step += 1

        info = self._get_info()
        info["expert_action"] = np.array([lin_vel, ang_vel], dtype=np.float32)

        observations = self._get_obs(info)
        reward, terminated, truncated = self._calculate_reward_and_termination(
            info, self.episode_time, observations, previous_obs
        )

        self.render()
        self.episode_return += reward
        info["episode_result"] = self.last_episode_result
        return observations, reward, terminated, truncated, info

    def _get_obs(self, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        current_target_distance, current_relative_angle, current_lidar = self._get_state()
        current_polar = np.array([current_target_distance, current_relative_angle], dtype=np.float32)
        self._update_lidar_stack(current_lidar)
        polar_stack = self._update_polar_stack(current_polar).reshape(self.n_stacking, 2)

        stacked_distances = polar_stack[:, 0]
        stacked_angles = polar_stack[:, 1]
        lidar_stack = self._get_stacked_lidar_obs()

        observation = np.concatenate([
            lidar_stack,
            stacked_distances,
            stacked_angles,
        ]).astype(np.float32)
        return observation

    def _update_polar_stack(self, polar_data: np.ndarray) -> np.ndarray:
        self.polar_stack.append(polar_data.copy())
        stacked = np.stack(list(self.polar_stack), axis=0)
        return stacked.flatten()

    def _get_info(self) -> Dict[str, Any]:
        target_sphere_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if target_sphere_id >= 0:
            sphere_pos = self.model.geom_pos[target_sphere_id]
            distance_to_sphere = np.linalg.norm(sphere_pos[:2] - self.robot_pos[:2])
        else:
            distance_to_sphere = float('inf')

        return {
            "distance_to_sphere": distance_to_sphere,
            "robot_position": self.robot_pos.copy(),
            "target_position": self.target_pos.copy(),
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "episode_count": self.episode_count,
            "current_step": self.current_step,
            "stacked_observations_shape": (self.n_stacking, self.num_rays),
            "total_observation_size": self.n_stacking * self.num_rays + self.n_stacking * 2,
            "scenario_id": self.current_scenario_id,
        }

    def _update_humans_simulation(self):
        if not self.all_obstacles:
            found_obstacles = self._get_obstacles_from_human_positions(self.humans_state_numpy)
            self.obstacles = self._get_static_obstacles_formatted(found_obstacles)

        self._update_human_goals()
        n_substeps = int(self.robot_dt / self.humans_dt)

        if self.manual_control:
            humans_state_jax = jnp.array(self.humans_state_numpy[:-1], dtype=jnp.float32)
            goals_jax = self.humans_current_goals[:-1]
            params_jax = self.human_parameters[:-1]
            obstacles_jax = self.obstacles[:-1] if self.obstacles is not None else None
            for _ in range(n_substeps):
                humans_state_jax = step(humans_state_jax, goals_jax, params_jax, obstacles_jax, self.humans_dt)
                if jnp.isnan(humans_state_jax).any():
                    logging.warning("NaN values detected in human state")
                    return
            self.humans_state_numpy[:-1] = np.array(humans_state_jax, dtype=np.float32)
            robot_id = self.mobile_robot_ID
            self.humans_state_numpy[-1, :2] = self.model.body_pos[robot_id, :2]
            self.humans_state_numpy[-1, 4] = self.data.qpos[2]
        else:
            humans_state_jax = jnp.array(self.humans_state_numpy, dtype=jnp.float32)
            goals_jax = self.humans_current_goals
            params_jax = self.human_parameters
            obstacles_jax = self.obstacles if self.obstacles is not None else None
            for _ in range(n_substeps):
                humans_state_jax = step(humans_state_jax, goals_jax, params_jax, obstacles_jax, self.humans_dt)
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
            human_id = self.human_body_ids[i]
            pos = self.humans_state_numpy[i, :2]
            orientation = self.humans_state_numpy[i, 4]
            self.model.body_pos[human_id, :] = [pos[0], pos[1], 0.0]
            half_angle = orientation / 2
            self.model.body_quat[human_id] = [np.cos(half_angle), 0., 0., np.sin(half_angle)]

        robot_id = self.mobile_robot_ID
        robot_pos = self.humans_state_numpy[-1, :2]
        robot_orientation = self.humans_state_numpy[-1, 4]
        self.model.body_pos[robot_id, :] = [robot_pos[0], robot_pos[1], 0.4]
        half_angle = robot_orientation / 2
        self.model.body_quat[robot_id] = [np.cos(half_angle), 0., 0., np.sin(half_angle)]

    def _calculate_reward_and_termination(self, info: Dict[str, Any], episode_time: float, obs, previous_obs) -> Tuple[float, bool, bool]:
        reward = 0.0
        terminated = False
        truncated = False

        # NOTE: gli indici qui seguono la tua struttura [dist_stack | angle_stack | lidar_stack]
        current_target_distance = obs[self.n_stacking - 1]
        current_relative_angle = obs[self.n_stacking * 2 - 1]
        current_lidar = obs[self.n_stacking * 2:]  # tutto il blocco lidar dopo i 2*n_stacking

        previous_distance = previous_obs[self.n_stacking - 1]

        progress_reward = PROGRESS_REWARD_SCALE * (previous_distance - current_target_distance)
        reward += progress_reward

        angle_reward = -0.01 * abs(current_relative_angle)
        reward += angle_reward

        if current_target_distance < DISTANCE_SUCCESS_THRESHOLD:
            self.last_episode_result = "success"
            terminated = True
            reward += 200
            return reward, terminated, truncated

        collision_detected, rew_lasers = self._lasers_reward(reward, current_lidar)
        reward += rew_lasers

        humans_collision_detected, _ = self.collision_detector.check_robot_human_collision_distance(
            collision_threshold=COLLISION_THRESHOLD
        )

        if collision_detected or humans_collision_detected:
            reward += -20 if humans_collision_detected else -10
            terminated = True

        if episode_time >= self.max_episode_time:
            self.last_episode_result = "timeout"
            truncated = True

        info.update({
            'episode_result': self.last_episode_result,
            'termination_reason': self.last_episode_result
        })
        return reward, terminated, truncated

    def _lasers_reward(self, base_reward: float, current_lidar) -> bool:
        collision_detected = False
        lidar_reward = 0.0
        for reading in current_lidar:
            if reading < COLLISION_THRESHOLD and reading > self.robot_radius:
                lidar_reward -= 0.1
            elif reading <= self.robot_radius:
                collision_detected = True
                return collision_detected, lidar_reward
        final_reward = base_reward + lidar_reward
        return collision_detected, final_reward

    @lru_cache(maxsize=128)
    def _get_obstacles_from_human_positions(self, humans_state_tuple: Tuple) -> List[List[str]]:
        humans_state = np.array(humans_state_tuple).reshape(-1, 6)
        obstacles_per_human = []
        grid_file_path = '/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid_cleaned.txt'
        try:
            with open(grid_file_path, 'r') as f:
                lines = f.readlines()
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
        except FileNotFoundError:
            logging.error("Grid file not found")
            return [[] for _ in range(self.n_humans)]
        return obstacles_per_human

    @lru_cache(maxsize=1)
    def _get_all_obstacles(self) -> Optional[jnp.ndarray]:
        mesh_file_path = '/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt'
        static_obstacles = []
        try:
            with open(mesh_file_path, 'r') as f:
                lines = f.readlines()
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
        except FileNotFoundError:
            logging.error("mesh_edges.txt not found")
            return None
        return jnp.array(static_obstacles) if static_obstacles else None

    def render(self, mode: str = 'human') -> bool:
        if rendering_disable:
            return False
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

    def get_observation_info(self) -> Dict[str, Any]:
        stacked_lidar_size = self.n_stacking * self.num_rays
        stacked_polar_size = self.n_stacking * 2
        return {
            "n_stacking": self.n_stacking,
            "num_rays": self.num_rays,
            "stacked_lidar_size": stacked_lidar_size,
            "stacked_polar_size": stacked_polar_size,
            "total_observation_size": stacked_lidar_size + stacked_polar_size,
            "observation_structure": {
                "stacked_lidar": f"indices 0 to {stacked_lidar_size - 1}",
                "stacked_distances": f"indices {stacked_lidar_size} to {stacked_lidar_size + self.n_stacking - 1}",
                "stacked_angles": f"indices {stacked_lidar_size + self.n_stacking} to {stacked_lidar_size + stacked_polar_size - 1}"
            },
            "lidar_stack_info": {
                "oldest_frame": f"indices 0 to {self.num_rays - 1}",
                "newest_frame": f"indices {(self.n_stacking - 1) * self.num_rays} to {self.n_stacking * self.num_rays - 1}"
            },
            "polar_stack_info": {
                "distances": f"shape ({self.n_stacking},) - oldest to newest",
                "angles": f"shape ({self.n_stacking},) - oldest to newest"
            }
        }

    def get_current_lidar_stack(self) -> np.ndarray:
        return np.stack(list(self.lidar_stack), axis=0)

    def _setup_viewer(self):
        # Import locale e lazy per evitare problemi in headless
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
