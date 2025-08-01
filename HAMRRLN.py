import numpy as np
import gymnasium as gym 
import mujoco
import mujoco.viewer
from mobilerobotRL import mobilerobotRL
import os
import xml.etree.ElementTree as ET
import time
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any

import logging
from collections import deque
from pynput import keyboard
# Import scenarios efficiently
from scenarios import scenario1, scenario1_easy, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12, scenario1_nohumans
#from no_humans_scenariosS import scenario1_nh, scenario2_nh, scenario3_nh, scenario4_nh, scenario5_nh, scenario6_nh, scenario7_nh, scenario8_nh, scenario9_nh, scenario10_nh, scenario11_nh, scenario12_nh

import jax.numpy as jnp
from JHSFM.jhsfm.hsfm import step
from JHSFM.jhsfm.utils import get_standard_humans_parameters
from grid_decomp.labeled_grid import GridCell_operations
from assets.collisondetector import CollisionDetector

from IL_HAMRRLN import NUM_RAYS, N_STACKING

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# Constants
ROBOT_RADIUS = 0.2
LIDAR_THRESHOLD = 0.4  # ROBOT_RADIUS * 2
HUMANS_DT = 0.01
DISTANCE_SUCCESS_THRESHOLD = 0.5

#NUM_RAYS = 108  # Number of lidar rays

MAX_EPISODE_TIME = 50 # MAX_EPISODE_TIME s

#N_STACKING = 10  # Default stacking size for observations

ROBOT_DT = 0.25 # Robot control timestep in seconds
MAX_LIN_VEL_ROBOT = 1.0    # da non confondere con il robot_dt che è il passo di controllo del robot
PROGRESS_REWARD_SCALE = 0.03  # Scale for progress reward

N_HUMANS = 5  # Default number of humans in the environment 

keyboard_active = False  # Set to False to disable keyboard control
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



class hamrrln(mobilerobotRL):
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

        
        
        # Timing parameters
        
        self.humans_dt = HUMANS_DT
        self.max_episode_time = MAX_EPISODE_TIME

        
        
        # Physics parameters
        self.robot_radius = ROBOT_RADIUS
        
        # Initialize humans-related components
        #self.human_parameters = get_standard_humans_parameters(self.n_humans + 1 if not self.training else self.n_humans)
        self.human_parameters = get_standard_humans_parameters(self.n_humans)

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
            scenario1_nohumans.scenario1_nohumans: 14,
        }

        self.scenario_successes = {i: 0 for i in range(1, len(self.scenario_mapping)+2)}
        self.scenario_attempts = {i: 1 for i in range(1, len(self.scenario_mapping)+2)}  # Avoid division by zero

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
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
        

        # Training scenarios
        # self.scenarios = [
        #     scenario1_easy.scenario1_easy,
        #     scenario4.scenario4, # Corridoio
        #     scenario9.scenario9, # Scenario con ostacoli    
        #     scenario10.scenario10, # Scenario con ostacoli e robot
        #     scenario11.scenario11, # Scenario con ostacoli e robot  
        #     scenario12.scenario12, # Scenario con ostacoli e robot
        #     scenario1_nohumans.scenario1_nohumans, # Scenario senza umani
        # ]

        # Testing scenarios for IL
        self.scenarios = [
            scenario1.scenario1, # uguale a scenario1 ma con target meno random
            scenario4.scenario4, # Corridoio
            scenario9.scenario9, # Scenario con ostacoli    
            scenario12.scenario12, # Scenario con ostacoli e robot
            scenario1.scenario1, # Scenario senza umani
            scenario11.scenario11, # Scenario con ostacoli e robot
            scenario7.scenario7, # Scenario con robot che gira tra le colonne
            scenario5.scenario5, # Scenario con robot che gira tra le colonne
            scenario6.scenario6, # Scenario con robot che gira tra le colonne
            scenario8.scenario8, # Scenario con robot che attraversa la porta con 3 umani nel mezzo
        ]
        
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
        self.all_obstacles = True
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=np.float32)
        
        if self.all_obstacles:
            obstacles_data = self._get_all_obstacles()
            if obstacles_data is not None:
                self.obstacles = jnp.stack([obstacles_data] * self.n_humans)
    
    def _reset_episode_counters(self):
        """Reset all episode-related counters."""
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

        self.progress_reward_scale_initial = 5.0
        self.progress_reward_scale_final = 0.03

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

    def _initialize_observation_stacking(self):
        """Initialize observation stacking components."""
        # Initialize deque to store lidar observations with fixed length
        self.lidar_stack = deque(maxlen=self.n_stacking)
        
        # Initialize with zeros (will be filled during first reset)
        empty_lidar = np.zeros(self.num_rays, dtype=np.float32)
        for _ in range(self.n_stacking):
            self.lidar_stack.append(empty_lidar.copy())
    
    def _setup_spaces(self):
        """Setup action and observation spaces with proper dtypes."""
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space depends on stacking mode
        if self.enable_stacking:
            # Stacked observations: [stacked_lidar_readings, stacked_distance_and_angle]
            stacked_lidar_size = self.num_rays * self.n_stacking
            stacked_polar_size = 2 * self.n_stacking  # distance and angle, each stacked n_stacking times
            
            obs_low = np.concatenate([
                np.zeros(stacked_lidar_size, dtype=np.float32),  # stacked lidar readings
                np.concatenate([
                    np.zeros(self.n_stacking, dtype=np.float32),     # stacked distances
                    np.full(self.n_stacking, -np.pi, dtype=np.float32)  # stacked angles
                ])
            ])
            
            obs_high = np.concatenate([
                np.full(stacked_lidar_size, 200.0, dtype=np.float32),  # stacked lidar readings
                np.concatenate([
                    np.full(self.n_stacking, 200.0, dtype=np.float32),  # stacked distances
                    np.full(self.n_stacking, np.pi, dtype=np.float32)   # stacked angles
                ])
            ])
            
            total_obs_size = stacked_lidar_size + stacked_polar_size
        else:
            # Non-stacked observations: [lidar_readings, distance, angle]
            obs_low = np.concatenate([
                np.zeros(self.num_rays, dtype=np.float32),  # lidar readings
                np.array([0.0, -np.pi], dtype=np.float32)   # distance and angle
            ])
            
            obs_high = np.concatenate([
                np.full(self.num_rays, 200.0, dtype=np.float32),  # lidar readings
                np.array([200.0, np.pi], dtype=np.float32)        # distance and angle
            ])
            
            total_obs_size = self.num_rays + 2
        
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
        
        # Cache human body IDs
        self.human_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            for i in range(self.n_humans)
        ], dtype=np.int32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        # Move robot position to (-2, -2)
        



        # Update episode statistics from previous episode
        if not self.training:
            #print(f"Episode return: {self.episode_return:.2f}")
            #print(f"Episode terminated after {self.current_step} steps or {self.current_step*self.robot_dt:.2f} SIMULATED seconds.")
            #print(f"Episode terminated after {self.robot_episode_steps} ROBOT steps or {self.robot_episode_steps*self.robot_dt:.2f} SIMULATED seconds."
            if self.episode_time > 0:
                elapsed_time = time.time() - self.start_time
        self._update_episode_stats()




        # Reset episode variables
        self.episode_count += 1
        self.current_step = 0
        self.previous_distance = 70.0
        self.episode_time_begin = time.time()
        self.episode_time = 0.0 
        self.last_episode_result = None
        self.relative_angle = 0.0
        self.robot_action_counter = 0 
        self.robot_episode_steps = 0
        self.robot_action_period = time.time()  # Reset action period timer, periodo tra un'azione e l'altra
        self.human_update_counter = 0

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

        
        # Initialize humans state tracking
        self._initialize_humans_tracking()

      

        self._reset_observation_stack() # Reset observation stack
        
        info = self._get_info()


        observation = self._get_obs(info)

        # print()
        # print()

        # print(f"Resetting environment with scenario ID: {self.current_scenario_id}") 
        mujoco.mj_forward(self.model, self.data)  # ensures all derived state like xmat is valid       
        return observation, info

    def _reset_observation_stack(self):
        """Reset the observation stack with the first lidar reading repeated."""
        self.lidar_stack.clear()
        self.polar_stack.clear()

        initial_robot_distance, initial_relative_angle, initial_lidar = self._get_state()  # Get initial state

        for _ in range(self.n_stacking):
            self.lidar_stack.append(initial_lidar.copy())

        empty_polar = np.array([initial_robot_distance, initial_relative_angle], dtype=np.float32)
        for _ in range(self.n_stacking):
            self.polar_stack.append(empty_polar.copy())
    
    def _update_lidar_stack(self, new_lidar_reading: np.ndarray):
        """Update the lidar stack with new reading."""
        # Add new reading (automatically removes oldest due to maxlen)
        if not self.enable_stacking:
            return

        self.lidar_stack.append(new_lidar_reading.copy())
    
    def _get_stacked_lidar_obs(self) -> np.ndarray:
        """Get stacked lidar observations as a flattened array."""
        if not self.enable_stacking:
            return self.lidar_readings.flatten()

        # Stack all lidar readings and flatten
        stacked = np.stack(list(self.lidar_stack), axis=0)  # Shape: (n_stacking, num_rays)
        return stacked.flatten()  # Shape: (n_stacking * num_rays,)
    
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
        sphere_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_id >= 0:
            target_x = scenario_data["target_robot_x"]
            target_y = scenario_data["target_robot_y"] 
            self.model.geom_pos[sphere_id, :] = [target_x, target_y, 2.0]
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
        # Get robot position and orientation
        """Get current observation with [goal info | distances | angles | lidar stack]."""
        #info = self._get_info()


        # Get current lidar readings
        current_lidar = np.zeros(self.num_rays, dtype=np.float32)
        for i, sensor_id in enumerate(self.lidar_sensor_ids):
            if sensor_id >= 0:
                current_lidar[i] = round(self.data.sensordata[sensor_id], 2)
            else:
                raise ValueError(f"⚠️ Invalid sensor ID for lidar_{i}: {sensor_id}")



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

        if self.training:
            self._apply_robot_action(action, dt = self.robot_dt)

        # Update humans simulation (robot_velocity_body now ready)
        self._update_humans_simulation(action)

        # Step MuJoCo physics
        mujoco.mj_step(self.model, self.data)

        # Get observation and info
        info = self._get_info()
        observation = self._get_obs(info)

        # Reward, done, truncated
        reward, terminated, truncated = self._calculate_reward_and_termination(info, self.episode_time, observation, prev_obs)
        self.episode_return += reward
        info["episode_result"] = self.last_episode_result

        return observation, reward, terminated, truncated, info



        
    
    def set_real_time_factor(self, factor: float):
        """Control how fast simulation runs relative to real-time.
        
        Args:
            factor: 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed
        """
        self.real_time_factor = max(0.01, factor)  # Prevent zero or negative
    
    def _get_obs(self, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Get current observation including stacked lidar readings."""

        
        current_target_distance, current_relative_angle, current_lidar = self._get_state()  # Get current state



        current_polar = np.array([current_target_distance, current_relative_angle], dtype=np.float32)

        self._update_lidar_stack(current_lidar)

        
        polar_stack = self._update_polar_stack(current_polar)  # shape (n_stacking * 2,)
        polar_stack = polar_stack.reshape(self.n_stacking, 2)

        # Separate distances and angles
        stacked_distances = polar_stack[:, 0]  # shape: (n_stacking,)
        stacked_angles = polar_stack[:, 1]     # shape: (n_stacking,)

        delta = self.target_pos[:2] - self.robot_pos[:2]
        target_angle = np.arctan2(delta[1], delta[0])

   

        # Get lidar stack
        lidar_stack = self._get_stacked_lidar_obs()  # shape: (n_stacking * num_rays,)

        # Final obs: [goal info] + [distances] + [angles] + [lidar]
        observation = np.concatenate([
            stacked_distances,
            stacked_angles,
            lidar_stack,
        ]).astype(np.float32)     

        return observation

    
    def _update_polar_stack(self, polar_data: np.ndarray) -> np.ndarray:
        """Update the polar stack with new data and return the flattened stack."""
        if not self.enable_stacking:
            return polar_data.flatten()

        self.polar_stack.append(polar_data.copy())
        # Return flattened stack for observation
        stacked = np.stack(list(self.polar_stack), axis=0)  # Shape: (n_stacking, 2)
        return stacked.flatten()  # Shape: (n_stacking * 2,)

    
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        # Calculate distance to sphere
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
            "total_observation_size": self.n_stacking * self.num_rays + self.n_stacking * 2,  # lidar + polar data
            "scenario_id": self.current_scenario_id,
        }
    
    def _apply_robot_action(self, action: np.ndarray, dt=ROBOT_DT):
        """Apply robot action using optimized kinematics."""
        self.action = action
        if not keyboard_active:

            max_lin_vel = MAX_LIN_VEL_ROBOT # Maximum linear velocity 0.25 m/s

            lin_vel = float(action[0])*max_lin_vel  # 0 to 1
            ang_vel = float(action[1])  # -1 to 1


            x, y, theta = self.data.qpos[:3]

    
            
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

            self.data.qpos[:3] = [x, y, theta]
            self.robot_pos = self.data.qpos[:2]  # Update robot position
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

        if not self.all_obstacles:
            # Dynamic obstacle detection (less efficient)
            found_obstacles = self._get_obstacles_from_human_positions(self.humans_state_numpy)
            self.obstacles = self._get_static_obstacles_formatted(found_obstacles)

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
                self.robot_pos[:] = self.data.xpos[self.mobile_robot_ID,:2]
                robot_theta = self.data.qpos[2]
                self.robot_theta = robot_theta  # save for use in humans_simulation

                # Compute robot velocity (in global frame)
                vx = (self.robot_pos[0] - self.prev_robot_pos[0]) / self.robot_dt
                vy = (self.robot_pos[1] - self.prev_robot_pos[1]) / self.robot_dt

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
                params_extended = jnp.concatenate([
                    self.human_parameters,
                    robot_params
                ], axis=0)

                # Ostacoli: aggiungi dummy per robot
                obstacles_extended = jnp.concatenate([
                    self.obstacles,
                    self.obstacles[:1]
                ], axis=0)

                

                

               

                # Esegui passo JHSFM e scarta il robot
                humans_state_with_robot = step(
                    humans_state_extended,
                    goals_extended,
                    params_extended,
                    obstacles_extended,
                    self.humans_dt,
                )

                # Apply robot action (NN output)
                self._apply_robot_action(action, dt = 0.01)
                humans_state_jax = humans_state_with_robot[:-1]  # escludi il robot
                mujoco.mj_step(self.model, self.data)  # Forward MuJoCo to update robot position
                
            else:
                # Training: nessun robot presente
                humans_state_jax = step(
                    humans_state_jax,
                    self.humans_current_goals,
                    self.human_parameters,
                    self.obstacles,
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

    def _calculate_reward_and_termination(self, info: Dict[str, Any], episode_time: float, obs, prev_obs) -> Tuple[float, bool, bool]:
        """Calculate reward and check for termination conditions."""
        reward = 0.0
        terminated = False
        truncated = False
        

        # Unpack obs
        current_lidar = obs[self.n_stacking*(2+self.num_rays-1):]  # current Lidar readings are the end of the observation array
        # Get distance to target and relative angle
        current_target_distance = obs[self.n_stacking-1]
        current_relative_angle = obs[self.n_stacking*2-1]


        previous_distance = prev_obs[self.n_stacking-1]

        # current_target_distance = obs[0]  # First element is distance to target
        # current_relative_angle = obs[1]  # Second element is relative angle
        # current_lidar = obs[2:]  # Remaining elements are lidar readings

        # previous_distance = prev_obs[0]  # First element of previous observation is distance to target


        # Adaptive progress reward scaling based on success rate
        success_rate = self.success_rate  # between 0 and 1
        if self.training:
            adaptive_scale = (1 - success_rate) * self.progress_reward_scale_initial + success_rate * self.progress_reward_scale_final
            progress_reward = adaptive_scale * (previous_distance - current_target_distance)
        else:
            progress_reward = self.progress_reward_scale_final * (previous_distance - current_target_distance)

        reward += progress_reward
        self.previous_distance = current_target_distance
        
        angle_reward = -0.01*abs(current_relative_angle)
        reward += angle_reward

        #print(f"Progress reward: {progress_reward:.5f}, Angle reward: {angle_reward:.5f}")

       

        # if not self.training:
        #    print(f"Distance to target: {distance_to_target:.2f}, Angle: {self.relative_angle:.2f}")
        
        # Collision detection and penalty
        collision_detected, rew_lasers = self._lasers_reward(reward, current_lidar)
        reward += rew_lasers
        
        # Success check
        if current_target_distance < DISTANCE_SUCCESS_THRESHOLD:
            self.last_episode_result = "success"
            terminated = True
            reward += 200
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = episode_time
            info["episode_result"] = "success"
            if not self.training:
                print(f"Target reached in {episode_time:.2f} seconds.")
                pass
            return reward, terminated, truncated
        
        # humans_collision_detected, _ = self.collision_detector.check_robot_human_collision_distance(
        #     collision_threshold=ROBOT_RADIUS
        # )

        if collision_detected:
            self.last_episode_result = "collision"
            terminated = True
            reward += -100
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = self.episode_time
            info["episode_result"] = "collision"
            return reward, terminated, truncated
        
        # Timeout check
        if self.robot_action_counter*self.robot_dt > self.max_episode_time:
            self.last_episode_result = "timeout"
            truncated = True
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = self.episode_time
            info["episode_result"] = "timeout"
            if not self.training:
                print(f"Episode timeout after {self.episode_time:.2f} REAL seconds.")
                pass
            return reward, terminated, truncated
        

        
        
        
        return reward, terminated, truncated
    
    def _lasers_reward(self, base_reward: float, current_lidar) -> bool:
        """Check for collisions and apply proximity penalties."""
        collision_detected = False
        base_reward = 0
        for i in range(0, len(current_lidar)):
            reading = current_lidar[i]
            collision_threshold = ROBOT_RADIUS
            
            if reading < LIDAR_THRESHOLD and reading > self.robot_radius: #0.4
                # Close to obstacle - apply penalty
                base_reward -= 0.1
            elif reading <= collision_threshold:
                # Collision detected
                collision_detected = True
                return collision_detected, base_reward
        
        return collision_detected, base_reward
























    @lru_cache(maxsize=128)
    def _get_obstacles_from_human_positions(self, humans_state_tuple: Tuple) -> List[List[str]]:
        """Get obstacles from human positions with caching."""
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
        """Load all obstacles from file with caching."""
        mesh_file_path = '/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt'
        static_obstacles = []
        
        try:
            with open(mesh_file_path, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.endswith(":"):
                    # Parse 4 vertices
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
                        # Create 4 edges for the obstacle
                        edges = [
                            [vertices[0], vertices[1]],
                            [vertices[1], vertices[2]], 
                            [vertices[2], vertices[3]],
                            [vertices[3], vertices[0]],
                        ]
                        static_obstacles.append(edges)
                    
                    i += 4  # Skip the vertex lines
                i += 1
                
        except FileNotFoundError:
            logging.error("mesh_edges.txt not found")
            return None
        
        return jnp.array(static_obstacles) if static_obstacles else None
    
    def render(self, mode: str = 'human') -> bool:
        """Render the environment."""
        if self.training:
            return False
        
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
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the observation structure.
        Helpful for understanding the stacking format.
        """
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
        """
        Get the current lidar stack as a 2D array for visualization/debugging.
        Returns shape: (n_stacking, num_rays)
        """
        return np.stack(list(self.lidar_stack), axis=0)