import numpy as np
import gymnasium as gym 
import mujoco
import mujoco.viewer
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
# Import scenarios efficiently
from scenarios import scenario1, scenario1_easy, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12
from scenarios import scenarioTEST1, scenarioTEST2, scenarioTEST3
#from no_humans_scenariosS import scenario1_nh, scenario2_nh, scenario3_nh, scenario4_nh, scenario5_nh, scenario6_nh, scenario7_nh, scenario8_nh, scenario9_nh, scenario10_nh, scenario11_nh, scenario12_nh

import jax
import jax.numpy as jnp
from JHSFM.jhsfm.hsfm import step as hsfm_step
from JHSFM.jhsfm.utils import get_standard_humans_parameters
from grid_decomp.labeled_grid import GridCell_operations
#from assets.collisondetector import CollisionDetector

from env_config import NUM_RAYS, N_STACKING, MAX_LIN_VEL_ROBOT, MAX_EPISODE_TIME, ROBOT_RADIUS, LIDAR_THRESHOLD, ROBOT_DT, HUMANS_DT, N_HUMANS, PROGRESS_REWARD_SCALE

THETA_HIST_LENGTH = 3
N_STACKING = 2

DEBUG = False
DEBUG_DATA = False

keyboard_active = os.environ.get('KEYBOARD_CONTROL', '0') == '1'
key_pressed = set()

def on_press(key):
    try: 
        key_pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        key_pressed.discard(key.char)
    except AttributeError:
        pass


# crea listener solo se esplicitamente richiesto e se esiste DISPLAY
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



class light_hamrrln(mobilerobotRL):
    """
    Optimized Human-Aware Mobile Robot Reinforcement Learning Navigation environment.
    Now includes observation stacking for temporal memory.
    """
    
    def __init__(self, 
                 num_rays: int = NUM_RAYS, 
                 model_path: str = "assets/world.xml", 
                 training: bool = True, 
                 n_humans: int = N_HUMANS, 
                 render_mode: Optional[str] = None,
                 n_stacking: int = N_STACKING,
                 enable_stacking: bool = True):
        
        # Core parameters
        self.num_rays = num_rays
        self.n_humans = n_humans
        self.training = training
        self.render_mode = render_mode 
        self.model_path = model_path
        self.n_stacking = n_stacking  # Number of observations to stack
        self.enable_stacking = enable_stacking  # Enable or disable observation stacking

        self.hsfm_step = jax.jit(hsfm_step)
        
        # Timing parameters
        
        self.humans_dt = HUMANS_DT
        self.max_episode_time = MAX_EPISODE_TIME

        
        
        # Physics parameters
        self.robot_radius = ROBOT_RADIUS
        
        # Initialize humans-related components
        #self.human_parameters = get_standard_humans_parameters(self.n_humans + 1 if not self.training else self.n_humans)
        self.human_parameters = get_standard_humans_parameters(self.n_humans)



        


        self.grid_cell_op = GridCell_operations(cell_size=4, world_size=320)
        self.max_grid_obs = 10
        self.max_edges = 4
        # dopo self.grid_cell_op = GridCell_operations(cell_size=4, world_size=320)
        self.grid_cell_op.load_labeled_grid_from_file(
            "/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid_cleaned.txt"
        )
        self.grid_cell_op.load_meshes_index(
            "/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt"
        )
        self.grid_radius = 1                   # 8 celle attorno (Moore neighborhood)
        self.use_grid_obstacles = True         # <— attiva schema veloce

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
            scenarioTEST3.scenarioTEST3: 16
        }

        self.scenario_successes = {i: 0 for i in range(1, len(self.scenario_mapping)+2)}
        self.scenario_attempts = {i: 1 for i in range(1, len(self.scenario_mapping)+2)}  # Avoid division by zero

        # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        # listener.start()
        
        # Pre-load obstacles for better performance
        self._initialize_obstacles()
        
        # Episode tracking
        self._reset_episode_counters()
        
        # Initialize state variables
        self._initialize_state_variables() 
        
        # Initialize observation stacking
        self._initialize_observation_stacking()
        
        # Define action and observation spaces with proper dtypes
        self._setup_spaces()
        

        self.scenarios = [
            ##########scenario1.scenario1, # Incrocio caos
            scenario4.scenario4, # Corridoio - PARALLEL TRAFFIC
            
            scenario5.scenario5, # Scenario con robot che si muove tra 3 tavoli con umani
            scenario6.scenario6, # Scenario con robot attravers DUE PORTE
            scenario7.scenario7, # Scenario con robot che gira tra le colonne
            scenario8.scenario8, # Scenario in fondo, con robot che attraversa la porta con 3 umani nel mezzo

            scenario9.scenario9, # PERPEDICULAR TRAFFIC - GENERALMENTE NON MESSO IN TRAINING

            scenario12.scenario12, # Scenario EASY, EASIEST VERY VERY VERY EASY stanza all'inizio a destra
            
            scenarioTEST2.scenarioTEST2, # TEST - Incrocio caos
            scenarioTEST3.scenarioTEST3, # TEST - PERPEDICULAR TRAFFIC con 7 umani
            scenarioTEST1.scenarioTEST1, # TEST - PARALLEL TRAFFIC

            # scenario 5 (robot tra 3 tavoli con umani che si muove parallelamente a lui )può essere usato ANCHE come test dopo training
        ]
        
        if len(self.scenarios) < 8:
            print("\n\n\n⚠️ STAI ADDESTRANDO CON POCHI SCENARI! ⚠️\n\n\n")


        self._setup_mujoco()


        self.robot_dt = ROBOT_DT
        
        # Setup viewer if needed
        self.viewer = None
        if self.render_mode == "human":
            self._setup_viewer()
        
        # Initialize environment
        self.reset()

    def _initialize_obstacles(self):
        """Initialize obstacle data efficiently."""
        self.obstacles = None
        self.all_obstacles = False  # Set to True to enable all obstacles !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=np.float32)

        #--- NEW: pre-carica linee raw (se disponibili) ---
        self._grid_lines = None
        self._mesh_lines = None
        try:
            with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid_cleaned.txt', 'r') as f:
                self._grid_lines = f.readlines()
        except FileNotFoundError:
            logging.error("Grid file not found (labeled_grid_cleaned.txt)")

        try:
            with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt', 'r') as f:
                self._mesh_lines = f.readlines()
        except FileNotFoundError:
            logging.error("Mesh file not found (mesh_edges.txt)")
        
        if self.all_obstacles:
            obstacles_data = self._get_all_obstacles()
            if obstacles_data is not None:
                self.obstacles = jnp.stack([obstacles_data] * self.n_humans)

    def _build_grid_obstacles_per_human(self):
        H = self.n_humans
        M = int(self.max_grid_obs)   # 10
        E = int(self.max_edges)      # 4
        out = np.full((H, M, E, 2, 2), np.nan, dtype=np.float32)  # padding a NaN

        for i in range(H):
            x = float(self.humans_state_numpy[i, 0])
            y = float(self.humans_state_numpy[i, 1])

            names = self.grid_cell_op.get_surrounding_obstacle_names_cached(
                x, y, radius=self.grid_radius, grid_size=60
            )
            # (n_obs_i, 4, 2, 2) – ogni ostacolo ha 4 spigoli (due punti 2D)
            obs_i = self.grid_cell_op.obstacles_for_names(names)

            if obs_i.size == 0:
                continue  # lascia NaN (HSFM dovrebbe ignorarli con il check isnan)

            # clip a massimo M ostacoli
            n = min(obs_i.shape[0], M)
            # se qualche mesh avesse <E spigoli, pad fino a E
            if obs_i.shape[1] < E:
                pad_edges = np.full((obs_i.shape[0], E - obs_i.shape[1], 2, 2), np.nan, dtype=np.float32)
                obs_i = np.concatenate([obs_i, pad_edges], axis=1)
            elif obs_i.shape[1] > E:
                obs_i = obs_i[:, :E]

            out[i, :n, :, :, :] = obs_i[:n]

        return jnp.array(out, dtype=jnp.float32)  # (H, 10, 4, 2, 2)





    
    def _reset_episode_counters(self):
        """Reset all episode-related counters at the beginning of the entire TRAINING"""
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
    
        
        # Rates (computed properties)
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
        self.real_time_factor = 1.0  # 1.0 means real-time


        if self.enable_stacking:
            self.polar_stack = deque([np.zeros(2, dtype=np.float32).copy() for _ in range(self.n_stacking)], maxlen=self.n_stacking)


    
    def _initialize_state_variables(self):
        """Initialize state tracking variables."""
        self.humans_goals = jnp.zeros((self.n_humans, 2, 2), dtype=jnp.float32)
        self.humans_current_goals = jnp.zeros((self.n_humans, 2), dtype=jnp.float32)
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.robot_rot_matrix = np.eye(3, dtype=np.float32)
        self.lidar_readings = np.zeros(self.num_rays, dtype=np.float32)
        self.action = np.zeros(2, dtype=np.float32)  # [linear_velocity, angular_velocity]
        self.robot_velocity_body = np.zeros(2, dtype=np.float32)  # [vx, vy]
        self.robot_theta = 0.0  # Robot orientation in radians

        self._theta_hist = deque(maxlen=THETA_HIST_LENGTH)

    def _initialize_observation_stacking(self):
        """Only keep the compact feature stack (t, t-1)."""
        from collections import deque
        self.lidar_feat_stack = deque(maxlen=2)

    
    def _setup_spaces(self):
        """Setup action and observation spaces with proper dtypes."""
        # Action space: allow gentle reverse and a narrower yaw range
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Compact obs: goal (dx,dy) + 2 frames of lidar features (3*K each)
        K = 24
        feat_per_frame = 3 * K  # min, mean, short-min for each sector
        total_obs_size = 2 + feat_per_frame * 2

        obs_low = np.concatenate([
            np.full(2, -np.inf, dtype=np.float32),               # dx, dy (can be large ±)
            np.zeros(feat_per_frame * 2, dtype=np.float32),      # features normalized to [0,1]
        ])
        obs_high = np.concatenate([
            np.full(2,  np.inf, dtype=np.float32),
            np.ones(feat_per_frame * 2, dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_obs_size,),
            dtype=np.float32
        )

    
    def _setup_mujoco(self):
        """Initialize MuJoCo model and data."""
        self.xml_model = self.load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)
        
        # Cache important IDs
        self.mobile_robot_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]
        
        # NEW: array NumPy fisso degli indici sensori (riusato a ogni step)
        self.sensor_ids_np = np.asarray(self.lidar_sensor_ids, dtype=np.int32)
        # (opzionale) valida che tutti i sensori siano stati trovati
        if np.any(self.sensor_ids_np < 0):
            missing = np.where(self.sensor_ids_np < 0)[0].tolist()
            logging.error(f"LIDAR sensors not found for indices: {missing}")

        # NEW: cache dell'ID del geom 'sphere' (bersaglio) per evitare lookup ripetuti
        self.sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if self.sphere_geom_id < 0:
            logging.error("Target geom 'sphere' not found in model.")

        # Cache human body IDs
        self.human_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            for i in range(self.n_humans)
        ], dtype=np.int32)





    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        
        
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
        self.robot_action_period = time.time()  # Reset action period timer, periodo tra un'azione e l'altra
        self.human_update_counter = 0
        self.progress_reward_term = 0.0
        self.laser_reward_term = 0.0
        self.progress_orientation_smoothness_term = 0.0

        # Reset previous action for smooth velocity tracking
        #self.previous_action = np.zeros(2, dtype=np.float32)

        # Save current position and orientation as reference for velocity computation
        robot_id = self.mobile_robot_ID
        self._prev_robot_pos = self.model.body_pos[robot_id, :2].copy() # serve la funzione _get_true_robot_velocities()

       
        


        mujoco.mj_forward(self.model, self.data)  # ensures all derived state like xmat is valid
        #self.robot_rot_matrix = self.data.body_xmat[robot_id].reshape(3, 3)
        
        
        # Call parent reset - this will call _get_obs() without info
        super().reset(seed=seed)
        
        # Load random scenario
        scenario_data = self._load_random_scenario() # load all the starting conditions from the scenario

        # Set humans initial states


        

        self._set_humans_initial_state(scenario_data) # Also place the target
        
        self._set_target_position(scenario_data)  # Set target sphere position

        self.robot_pos = np.array([scenario_data["mob_robot_startposx"], scenario_data["mob_robot_startposy"]])
        self.robot_theta = scenario_data["mob_robot_start_orientation"]
        self.data.qpos[:3] = [self.robot_pos[0], self.robot_pos[1], self.robot_theta]

        self._theta_hist.clear()
        for _ in range(THETA_HIST_LENGTH):
            self._theta_hist.append(self.robot_theta)

        
        # Initialize humans state tracking
        self._initialize_humans_tracking()
        
        info = self._get_info()


        observation = self._get_obs(info)

        # print()
        # print()

        # print(f"Resetting environment with scenario ID: {self.current_scenario_id}") 
        mujoco.mj_forward(self.model, self.data)  # ensures all derived state like xmat is valid       
        return observation, info




    

    # --- Compact LiDAR feature extraction (3*K features) ---
    def _lidar_to_features(self, lidar: np.ndarray, k: int = 24, max_range: float = 10.0, short_r: float = 1.0) -> np.ndarray:
        """Convert raw lidar rays to compact sector features (min, mean, short-range min)."""
        rays = int(self.num_rays)
        bins = np.array_split(lidar[:rays], k)

        feat_min  = np.array([np.min(b) for b in bins], dtype=np.float32)
        feat_mean = np.array([np.mean(b) for b in bins], dtype=np.float32)

        short_bins = []
        for b in bins:
            sel = b[b <= short_r]
            if sel.size == 0:
                short_bins.append(np.array([max_range], dtype=np.float32))
            else:
                short_bins.append(sel)
        feat_smin = np.array([np.min(b) for b in short_bins], dtype=np.float32)

        clip01 = lambda x: np.clip(x / max_range, 0.0, 1.0)
        return np.concatenate([clip01(feat_min), clip01(feat_mean), clip01(feat_smin)]).astype(np.float32)

    def _goal_egocentric(self) -> np.ndarray:
        """Return goal vector in robot frame: (dx, dy)."""
        dxg = float(self.target_pos[0] - self.robot_pos[0])
        dyg = float(self.target_pos[1] - self.robot_pos[1])
        ct, st = np.cos(self.robot_theta), np.sin(self.robot_theta)
        dx =  ct * dxg + st * dyg
        dy = -st * dxg + ct * dyg
        return np.array([dx, dy], dtype=np.float32)

    
    def _update_episode_stats(self):
        """Update episode statistics."""
        if self.current_step > 0 and self.last_episode_result:
            if self.last_episode_result == "success":
                self.success_count += 1
            elif self.last_episode_result == "collision":
                self.collision_count += 1
            elif self.last_episode_result == "human_collision":
                self.human_collision_count += 1
            elif self.last_episode_result == "timeout":
                self.timeout_count += 1

            # Update per-scenario success tracking
            if self.current_scenario_id is not None:
                self.scenario_attempts[self.current_scenario_id] += 1
                if self.last_episode_result == "success":
                    self.scenario_successes[self.current_scenario_id] += 1
            
            # Calculate rates
            if self.episode_count > 1:
                total_episodes = self.episode_count - 1
                self.success_rate = self.success_count / total_episodes
                self.collision_rate = self.collision_count / total_episodes
                self.timeout_rate = self.timeout_count / total_episodes
                
                # Log results for evaluation
                if not self.training:
                    self._log_episode_result(total_episodes)

    
    def _log_episode_result(self, episode_num: int):
        """Log episode results during evaluation."""
        result_msg = (f"{self.last_episode_result.upper()}: "
                     f"Episode={episode_num} sr={self.success_rate:.2f}, "
                     f"cr={self.collision_rate:.2f},tr={self.timeout_rate:.2f}, ")
        print(result_msg)
    
    def _load_random_scenario(self) -> Dict[str, float]:
        """Load a random scenario with probability inversely proportional to its success rate."""
        if not self.training: # during evaluation choose scenarios randomly using uniform distribution
            scenario_func = random.choice(self.scenarios)
            self.current_scenario_id = self.scenario_mapping[scenario_func]
            return scenario_func()

        # Compute per-scenario success rates
        weights = []
        for func in self.scenarios:
            sid = self.scenario_mapping[func]
            success_rate = self.scenario_successes[sid] / self.scenario_attempts[sid]
            weight = 1.0 - success_rate  # Higher success = lower weight
            weights.append(max(weight, 0.01))  # Avoid zero probability

        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()

        # Sample scenario based on weights
        scenario_func = random.choices(self.scenarios, weights=weights, k=1)[0]
        self.current_scenario_id = self.scenario_mapping[scenario_func]
        return scenario_func()


    
    def _set_robot_initial_state(self, scenario_data: Dict[str, float]):
        """Set robot initial position and orientation."""
        self.data.qpos[0] = scenario_data["mob_robot_startposx"]
        self.data.qpos[1] = scenario_data["mob_robot_startposy"] 
        self.data.qpos[2] = scenario_data["mob_robot_start_orientation"]
        self.data.qvel[:] = np.zeros_like(self.data.qvel)

    
    def _set_humans_initial_state(self, scenario_data: Dict[str, float]):
        """Set humans initial positions and goals efficiently."""
        # Extract human data more efficiently
        human_positions = []
        human_goals = []
        
        for i in range(1, self.n_humans + 1):
            start_pos = [scenario_data[f"human{i}x"], scenario_data[f"human{i}y"]]
            target_pos = [scenario_data[f"targethuman{i}x"], scenario_data[f"targethuman{i}y"]]
            orientation = scenario_data[f"start_orientation_human{i}"]
            
            human_positions.append(start_pos)
            human_goals.append([start_pos, target_pos])
            
            # Set position and orientation in MuJoCo
            human_id = self.human_body_ids[i-1]
            self.model.body_pos[human_id, :2] = start_pos
            self.model.body_quat[human_id] = [
                np.cos(orientation/2), 0., 0., np.sin(orientation/2)
            ]
        
        self.humans_goals = jnp.array(human_goals, dtype=jnp.float32)
        self.humans_current_goals = jnp.array([pos for pos, _ in human_goals], dtype=jnp.float32)
    
    def _set_target_position(self, scenario_data: Dict[str, float]):
        """Set target sphere position."""
        if self.sphere_geom_id >= 0:
            target_x = scenario_data["target_robot_x"]
            target_y = scenario_data["target_robot_y"] 
            self.model.geom_pos[self.sphere_geom_id, :] = [target_x, target_y, 2.0]
        self.target_pos = np.array([scenario_data["target_robot_x"], scenario_data["target_robot_y"]], dtype=np.float32)
    
    def _initialize_humans_tracking(self):
        """Initialize humans state tracking arrays."""
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=np.float32)
        
        for i in range(self.n_humans):
            human_id = self.human_body_ids[i]
            # Position
            self.humans_state_numpy[i, :2] = self.model.body_pos[human_id, :2]
            # Velocity (initially zero)
            self.humans_state_numpy[i, 2:4] = 0.0
            # Orientation from quaternion
            w, x, y, z = self.model.body_quat[human_id] 
            theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            self.humans_state_numpy[i, 4] = theta
            # Angular velocity (initially zero)
            self.humans_state_numpy[i, 5] = 0.0

        
    def _get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        current_lidar = np.asarray(self.data.sensordata[self.sensor_ids_np], dtype=np.float32)

        # --- GOAL INFO ---
        delta = self.target_pos[:2] - self.robot_pos[:2]
        distance_to_target = np.linalg.norm(delta)
        target_angle = np.arctan2(delta[1], delta[0])

        robot_orientation = self.robot_theta  # Use robot_theta directly

        relative_angle = target_angle - robot_orientation
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        self.relative_angle = relative_angle  

        #print(f"Distance to target: {distance_to_target:.2f}, Relative angle: {(relative_angle):.2f}")  # Debug info
        # print lidar
        
        return distance_to_target, relative_angle, current_lidar



































    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        prev_obs = self._get_obs() 

        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"

        if not self.training: # in testing
            # Enforce real-time stepping in evaluation mode
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

        if self.training: # if training, robot is not seen by humans
            self._apply_robot_action(action, dt = self.robot_dt)

        # Update humans simulation (robot_velocity_body now ready)
        self._update_humans_simulation(action)

        # Step MuJoCo physics
        mujoco.mj_step(self.model, self.data)




        self._theta_hist.append(float(self.robot_theta))

        # Get observation and info after having applied action and stepped physics
        info = self._get_info()
        observation = self._get_obs(info)

        # Reward, done, truncated
        if DEBUG:
            step_reward, terminated, truncated, progress_reward_term, laser_reward_term, theta_smoothness_term = self._calculate_reward_and_termination(info, self.episode_time, observation, prev_obs)
        else:
            step_reward, terminated, truncated = self._calculate_reward_and_termination(info, self.episode_time, observation, prev_obs)
        self.episode_return += step_reward
        info["episode_result"] = self.last_episode_result
        if terminated or truncated:
            
            if DEBUG:
                print(f"Episode return: {self.episode_return}")
                print(f"Progress reward term: {progress_reward_term}")
                print(f"Laser reward term: {laser_reward_term}")
                print(f"Theta smoothness term: {theta_smoothness_term}")

        return observation, step_reward, terminated, truncated, info



        
    
    def set_real_time_factor(self, factor: float):
        """Control how fast simulation runs relative to real-time.
        
        Args:
            factor: 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed
        """
        self.real_time_factor = max(0.01, factor)  # Prevent zero or negative
    
    def _get_obs(self, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        New compact observation:
        [ dx, dy, lidar_features(t), lidar_features(t-1) ]
        where lidar_features = [sector_min, sector_mean, short_min] for K sectors.
        """
        # Pull current raw state (and keeps internal caches updated)
        current_target_distance, current_relative_angle, current_lidar = self._get_state()

        # Goal as egocentric vector
        goal_vec = self._goal_egocentric()  # shape (2,)

        # Current features
        feats_t = self._lidar_to_features(current_lidar, k=24, max_range=10.0, short_r=1.0)
        # Maintain a two-frame stack
        if len(self.lidar_feat_stack) == 0:
            self.lidar_feat_stack.append(feats_t.copy())
        self.lidar_feat_stack.append(feats_t.copy())
        if len(self.lidar_feat_stack) < 2:
            feats_tm1 = feats_t
        else:
            feats_tm1 = self.lidar_feat_stack[-2]

        observation = np.concatenate([goal_vec, feats_t, feats_tm1]).astype(np.float32)
        return observation


    
    
    def _get_info(self) -> Dict[str, Any]:
        """Compact info object matching the new observation format."""
        # Distance to target sphere (same as before)
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
            # Size hint for the new obs: [dx, dy] + feats(t) + feats(t-1) = 146 for K=24 (3*K per frame * 2 frames + 2)
            "total_observation_size": 146,
        }

    
    def _collision_detection(self, current_lidar):
        
        if np.min(current_lidar) <= ROBOT_RADIUS:
            if DEBUG:
                print(f"Min LIDAR: {np.min(current_lidar):.3f}")
            return True

        return False
    
    def _apply_robot_action(self, action: np.ndarray, dt=ROBOT_DT):
        """Apply robot action using optimized kinematics."""
        self.action = action
        if not keyboard_active:

            max_lin_vel = MAX_LIN_VEL_ROBOT # Maximum linear velocity 0.25 m/s

            lin_vel = float(action[0])*max_lin_vel  # 0 to 1
            ang_vel = float(action[1])  # -1 to 1


            #x, y, theta = self.data.qpos[:3]
            x, y, theta = self.robot_pos[0], self.robot_pos[1], self.robot_theta
    
            
            # Optimized kinematic update
            if abs(ang_vel) > 1e-3:
                # Non-zero angular velocity - use circular arc motion
                ratio = lin_vel / ang_vel
                sin_new = np.sin(theta + ang_vel * dt)
                sin_old = np.sin(theta)
                cos_new = np.cos(theta + ang_vel * dt)
                cos_old = np.cos(theta)
                
                x += ratio * (sin_new - sin_old)
                y += ratio * (-cos_new + cos_old)
                theta += ang_vel * dt
            else:
                # Zero angular velocity - straight line motion
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                x += lin_vel * cos_theta * dt
                y += lin_vel * sin_theta * dt

            # Warp theta to [-pi, pi]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            self.data.qpos[:3] = np.array([x, y, theta], dtype=np.float32)
            self.robot_pos = np.array([x, y], dtype=np.float32)  # Update robot position
            self.robot_theta = theta  # Update robot orientation

        else:

            # Keyboard control for manual testing

            step_size = 0.1  # Movement step size
            rotation_step = 0.1  # Rotation step size

            # Get the current yaw angle (orientation) of the agent
            yaw = self.data.qpos[2]

            # Calculate the forward direction based on the yaw angle
            forward_x = np.cos(yaw)
            forward_y = np.sin(yaw)

            # Move forward or backward in the facing direction
            if 'u' in key_pressed:  # Move forward
                self.data.qpos[0] += step_size * forward_x
                self.data.qpos[1] += step_size * forward_y
            if 'n' in key_pressed:  # Move backward
                self.data.qpos[0] -= step_size * forward_x
                self.data.qpos[1] -= step_size * forward_y

            # Rotate the agent
            if 'g' in key_pressed:  # Rotate clockwise
                self.data.qpos[2] += rotation_step
            if 'k' in key_pressed:  # Rotate counterclockwise
                self.data.qpos[2] -= rotation_step
            
            
        
    
    def _update_humans_simulation(self, action):
        """Update humans simulation using HSFM, optionally including robot in the simulation during testing."""

        # Build obstacles for each human around the grid
        if self.use_grid_obstacles:
            obstacles_agents = self._build_grid_obstacles_per_human()
        else:
            # fallback legacy: TUTTI gli ostacoli uguali per ogni umano (costoso)
            if self.all_obstacles and self.obstacles is not None:
                obstacles_agents = self.obstacles
            else:
                # vecchio metodo (I/O pesante) — sconsigliato
                found = self._get_obstacles_from_human_positions(self.humans_state_numpy)
                obstacles_agents = self._get_static_obstacles_formatted(found)


        # Update human goals if targets are reached
        self._update_human_goals()

        # Convert current numpy state to JAX array
        humans_state_jax = jnp.array(self.humans_state_numpy, dtype=jnp.float32)

        # Number of substeps (25) for 0.25s / 0.01s
        n_substeps = int(self.robot_dt / self.humans_dt)

        for _ in range(n_substeps):
            if not self.training:
                # Costruisci lo stato del robot da aggiungere temporaneamente
                # Track previous robot position
                self.prev_robot_pos = self.robot_pos.copy()
                self.robot_pos[:] = self.data.qpos[:2]
                robot_theta = self.data.qpos[2]
                self.robot_theta = robot_theta  # save for use in humans_simulation

                # Compute robot velocity (in global frame)
                vx = (self.robot_pos[0] - self.prev_robot_pos[0]) / self.humans_dt
                vy = (self.robot_pos[1] - self.prev_robot_pos[1]) / self.humans_dt

                # Compute velocity in robot body frame
                rot_matrix = np.array([
                    [np.cos(robot_theta), np.sin(robot_theta)],
                    [-np.sin(robot_theta), np.cos(robot_theta)]
                ])
                self.robot_velocity_body = rot_matrix @ np.array([vx, vy])  # shape (2,)
                robot_state = jnp.array([
                    self.robot_pos[0], self.robot_pos[1],
                    self.robot_velocity_body[0], self.robot_velocity_body[1],
                    self.robot_theta, 0.0
                ], dtype=jnp.float32)

                # Costruisci lo stato completo estendendo con il robot
                humans_state_extended = jnp.concatenate([
                    humans_state_jax,
                    robot_state[None, :]
                ], axis=0)

                # Obiettivi: umani + robot (verso il target)
                robot_goal = jnp.array(self.target_pos[:2], dtype=jnp.float32)
                goals_extended = jnp.concatenate([
                    self.humans_current_goals,
                    robot_goal[None, :]
                ], axis=0)

                # Parametri: umani + copia di un umano per il robot
                robot_params = self.human_parameters[:1]  # oppure un set dedicato
               #robot_params = robot_params.at[0].set(ROBOT_RADIUS) 
                params_extended = jnp.concatenate([
                    self.human_parameters,
                    robot_params
                ], axis=0)

         
                obstacles_extended = jnp.concatenate([obstacles_agents, obstacles_agents[:1]], axis=0)
                humans_state_with_robot = self.hsfm_step(
                    humans_state_extended,
                    goals_extended,
                    params_extended,
                    obstacles_extended,
                    self.humans_dt,
                )


                # Apply robot action (NN output)
                self._apply_robot_action(action, dt = 0.01)
                humans_state_jax = humans_state_with_robot[:-1]  # escludi il robot
                
            else:
                # Training: nessun robot presente
                humans_state_jax = self.hsfm_step(
                    humans_state_jax,
                    self.humans_current_goals,
                    self.human_parameters,
                    obstacles_agents,
                    self.humans_dt,
                )

        # Controllo NaN
        if jnp.isnan(humans_state_jax).any():
            logging.warning("NaN values detected in human state")
            return

        # Salva risultato finale
        self.humans_state_numpy = np.array(humans_state_jax, dtype=np.float32)

        # Aggiorna le posizioni su MuJoCo
        self._update_human_positions_in_mujoco()

    
    def _update_human_goals(self):
        """Update human goals when targets are reached."""
        for i in range(self.n_humans):
            current_pos = self.humans_state_numpy[i, :2]
            current_goal = self.humans_current_goals[i]
            distance_to_goal = np.linalg.norm(current_pos - current_goal)
            
            if distance_to_goal < 0.1:
                # Check which goal is currently active and switch
                goal_0 = self.humans_goals[i, 0]
                goal_1 = self.humans_goals[i, 1]
                
                if np.allclose(current_goal, goal_0, atol=1e-6):
                    self.humans_current_goals = self.humans_current_goals.at[i].set(goal_1)
                else:
                    self.humans_current_goals = self.humans_current_goals.at[i].set(goal_0)
    
    def _update_human_positions_in_mujoco(self):
        """Update human positions and orientations in MuJoCo simulation."""
        for i in range(self.n_humans):
            human_id = self.human_body_ids[i]
            pos = self.humans_state_numpy[i, :2]
            orientation = self.humans_state_numpy[i, 4]
            
            # Update position
            self.model.body_pos[human_id, :] = [pos[0], pos[1], 0.0]
            
            # Update orientation
            half_angle = orientation / 2
            self.model.body_quat[human_id] = [
                np.cos(half_angle), 0., 0., np.sin(half_angle)
            ]


    def _ang_diff(self, a: float, b: float) -> float:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    
    # --- utils (metti vicino ad altre util se preferisci) ---
    def _safe_div(self, num: float, den: float, eps: float = 1e-8) -> float:
        return float(num) / float(den + eps)

    def _get_max_ang_vel(self) -> float:
        # Usa una costante se già esiste altrove; altrimenti un fallback sensato
        return float(getattr(self, "MAX_ANG_VEL_ROBOT", 1.5))


    # --- LIDAR proximity reward (stabile e normalizzato) ---
    def _lasers_reward(self, current_lidar) -> float:
        """Penalty crescente quando l'ostacolo più vicino entra sotto soglia."""
        min_lidar = float(np.min(current_lidar))
        if min_lidar >= LIDAR_THRESHOLD:
            return 0.0

        # Normalizza la vicinanza in [0,1]: 0 = sicuro, 1 = a contatto col raggio robot
        denom = max(1e-6, (LIDAR_THRESHOLD - float(self.robot_radius)))  # epsilon
        proximity = np.clip((LIDAR_THRESHOLD - min_lidar) / denom, 0.0, 1.0)
        # Peso lidar (tuning): 1.0 dà segnali chiari ma non devastanti
        w_lidar = 1.0
        return -w_lidar * float(proximity)


    # --- Smoothness su θ con normalizzazione fisica ---
    def _orientation_smoothness_penalty(self) -> float:
        """
        Penalizza variazioni veloci di orientazione (dθ) e jerk (ddθ),
        normalizzando rispetto al limite fisico di vel. angolare.
        """
        if not hasattr(self, "_theta_hist") or len(self._theta_hist) < 2:
            return 0.0

        dt = float(self.robot_dt)
        th = list(self._theta_hist)  # [..., θ_{t-2}, θ_{t-1}, θ_t]
        max_w = self._get_max_ang_vel()

        # dθ_t (rad/s)
        dtheta_t = self._ang_diff(th[-1], th[-2]) / dt
        dtheta_n = dtheta_t / (max_w + 1e-8)

        penalty = 0.0
        lam_d, lam_dd = 0.02, 0.01  # pesi moderati e interpretabili

        penalty += -lam_d * (dtheta_n ** 2)

        if len(th) >= 3:
            dtheta_tm1 = self._ang_diff(th[-2], th[-3]) / dt
            ddtheta_t = (dtheta_t - dtheta_tm1) / dt           # rad/s^2
            ddtheta_n = ddtheta_t / (max_w / max(dt, 1e-8))    # normalizza rispetto alla scala fisica
            penalty += -lam_dd * (ddtheta_n ** 2)

        # Clip prudente per evitare outlier rari (es. spawn / teletrasporti)
        return float(np.clip(penalty, -0.3, 0.0))


    # --- Reward principale e terminazioni ---
    def _calculate_reward_and_termination(self, info: Dict[str, Any], episode_time: float, obs, prev_obs):
        step_reward = 0.0
        terminated, truncated = False, False

        # Stato corrente
        current_target_distance, current_relative_angle, current_lidar = self._get_state()  # :contentReference[oaicite:0]{index=0}

        # 1) Collisione immediata
        if self._collision_detection(current_lidar):
            
            self.last_episode_result = "collision"
            terminated = True
            step_reward += -70.0
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = self.episode_time
            info["episode_result"] = "collision"
            if DEBUG:
                # inizializza termini se non presenti
                if not hasattr(self, "progress_reward_term"): self.progress_reward_term = 0.0
                if not hasattr(self, "laser_reward_term"): self.laser_reward_term = 0.0
                if not hasattr(self, "progress_orientation_smoothness_term"): self.progress_orientation_smoothness_term = 0.0
                if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term
            return step_reward, terminated, truncated

        # 2) Progress (normalizzato e stabile)
        # Opzione adattiva clampata (se vuoi mantenerla)
        if self.training:
            sr = float(getattr(self, "success_rate", 0.0))
            s0 = float(getattr(self, "progress_reward_scale_initial", 0.04))
            s1 = float(getattr(self, "progress_reward_scale_final",   0.08))
            adaptive_scale = np.clip((1 - sr) * s0 + sr * s1, 0.02, 0.12)
            progress_reward = adaptive_scale * (float(self.previous_distance) - float(current_target_distance))
        else:
            progress_reward = 0.06 * (float(self.previous_distance) - float(current_target_distance))

        # Clip per-step per evitare spike (es. spawn vicino al goal)
        progress_reward = float(np.clip(progress_reward, -0.2, 0.2))
        step_reward += progress_reward

        # 3) Allineamento angolare (normalizzato su π)
        angle_norm = abs(float(current_relative_angle)) / np.pi  # in [0,1]
        angle_reward = -0.05 * angle_norm
        angle_reward = float(np.clip(angle_reward, -0.1, 0.0))
        step_reward += angle_reward

        # 4) LIDAR proximity (già normalizzato)
        lidar_reward = self._lasers_reward(current_lidar)
        lidar_reward = float(np.clip(lidar_reward, -1.0, 0.0))
        step_reward += lidar_reward

        # 5) Smoothness orientazionale (θ)
        theta_smoothness_penalty = self._orientation_smoothness_penalty()
        theta_smoothness_penalty = float(np.clip(theta_smoothness_penalty, -0.3, 0.0))
        step_reward += theta_smoothness_penalty


        # 6) Success
        if current_target_distance <= ROBOT_RADIUS:
            #print(current_target_distance)
            self.last_episode_result = "success"
            terminated = True
            step_reward += 200.0
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = episode_time
            info["episode_result"] = "success"
            # aggiorna previous_distance solo a fine step/terminazione
            self.previous_distance = float(current_target_distance)
            if DEBUG:
                if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                self.progress_reward_term += progress_reward
                self.laser_reward_term += lidar_reward
                self.progress_orientation_smoothness_term += theta_smoothness_penalty
                self.angle_reward_term += angle_reward
                return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term
            return step_reward, terminated, truncated

        # 7) Timeout (misurato in sim-time)
        if self.robot_action_counter * self.robot_dt > self.max_episode_time:
            self.last_episode_result = "timeout"
            truncated = True
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = self.episode_time
            info["episode_result"] = "timeout"
            self.previous_distance = float(current_target_distance)
            if DEBUG:
                if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
                self.progress_reward_term += progress_reward
                self.laser_reward_term += lidar_reward
                self.progress_orientation_smoothness_term += theta_smoothness_penalty
                self.angle_reward_term += angle_reward
                return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term
            return step_reward, terminated, truncated

        # 8) Aggiorna previous_distance SOLO qui (episodio continua)
        self.previous_distance = float(current_target_distance)

        if DEBUG:
            if not hasattr(self, "angle_reward_term"): self.angle_reward_term = 0.0
            self.progress_reward_term += progress_reward
            self.laser_reward_term += lidar_reward
            self.progress_orientation_smoothness_term += theta_smoothness_penalty
            self.angle_reward_term += angle_reward
            return step_reward, terminated, truncated, self.progress_reward_term, self.laser_reward_term, self.progress_orientation_smoothness_term

        return step_reward, terminated, truncated
























    @lru_cache(maxsize=128)
    def _get_obstacles_from_human_positions(self, humans_state_tuple: Tuple) -> List[List[str]]:
        """Get obstacles from human positions with caching."""
        humans_state = np.array(humans_state_tuple).reshape(-1, 6)
        obstacles_per_human = []

        # --- NEW: usa le linee pre-caricate, se disponibili ---
        lines = self._grid_lines
        if lines is None:
            # Fallback legacy (identico al tuo comportamento attuale)
            grid_file_path = '/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid_cleaned.txt'
            try:
                with open(grid_file_path, 'r') as f:
                    lines = f.readlines()
            except FileNotFoundError:
                logging.error("Grid file not found")
                return [[] for _ in range(self.n_humans)]

        # parsing identico all’attuale
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
        """Load all obstacles from file with caching."""
        static_obstacles = []

        # --- NEW: usa linee pre-caricate se presenti ---
        lines = self._mesh_lines
        if lines is None:
            mesh_file_path = '/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt'
            try:
                with open(mesh_file_path, 'r') as f:
                    lines = f.readlines()
            except FileNotFoundError:
                logging.error("mesh_edges.txt not found")
                return None

        # parsing identico al tuo (blocchi da 1 riga titolo + 4 coordinate)
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

                i += 4  # salta le 4 righe di coordinate
            i += 1

        return jnp.array(static_obstacles) if static_obstacles else None

    
    def render(self, mode: str = 'human') -> bool:
        """Render the environment."""
        
        if self.viewer is None:
            self._setup_viewer()
        
        if mode == 'human' and self.viewer is not None:
            self.viewer.sync()
            #time.sleep(0.01)
            return True
        
        return False
    
    def close(self):
        """Clean up resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
