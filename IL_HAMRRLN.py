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
# from pynput import keyboard
# Import scenarios efficiently
from scenarios import scenario1, scenario1_easy, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12, scenario1_nohumans
#from no_humans_scenariosS import scenario1_nh, scenario2_nh, scenario3_nh, scenario4_nh, scenario5_nh, scenario6_nh, scenario7_nh, scenario8_nh, scenario9_nh, scenario10_nh, scenario11_nh, scenario12_nh

import jax.numpy as jnp
from JHSFM.jhsfm.hsfm import step
from JHSFM.jhsfm.utils import get_standard_humans_parameters
from grid_decomp.labeled_grid import GridCell_operations
from assets.collisondetector import CollisionDetector

from pynput import keyboard
import glfw

keyboard_active = False  # Set to False to disable keyboard control
key_pressed = set()
TOGGLE_KEY  = 'm'     # press to switch HSFM ↔ keyboard
FWD_KEY     = 'u'
BACK_KEY    = 'n'
LEFT_KEY    = 'k'     # CCW  (to match your old code)
RIGHT_KEY   = 'g'     # CW


import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# Constants
ROBOT_RADIUS = 0.2
COLLISION_THRESHOLD = 0.4  # ROBOT_RADIUS * 2
DISTANCE_SUCCESS_THRESHOLD = 0.7
MAX_EPISODE_TIME = 100.0 # MAX_EPISODE_TIME s
HUMANS_DT = 0.01
N_STACKING = 10  # Default stacking size for observations
ROBOT_DT = 0.25 # Robot control timestep in seconds
MAX_LIN_VEL_ROBOT = 1.0     # da non confondere con il robot_dt che è il passo di controllo del robot
PROGRESS_REWARD_SCALE = 0.03  # Scale for progress reward
REPELLENT_FORCE = 0.35
REPELLENT_WALL_FORCE = 1.0

NUM_RAYS = 108


rendering_disable = True
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



class il_hamrrln(mobilerobotRL):
    """
    The same HAMRRLN environment as before, but now supports Imitation Learning
    The robot will now move as a human using HSFM as the human model.
    """
    
    def __init__(self, 
                 num_rays: int = NUM_RAYS, 
                 model_path: str = "assets/world.xml", 
                 training: bool = True, 
                 n_humans: int = 5, 
                 render_mode: Optional[str] = None,
                 n_stacking: int = N_STACKING):
        
        # Core parameters
        self.num_rays = num_rays
        self.n_humans = n_humans
        self.training = training
        self.render_mode = None
        self.model_path = model_path
        self.n_stacking = n_stacking  # Number of observations to stack
        self._prev_robot_pos = np.zeros(2, dtype=np.float32)  # serviva per la funzione _get_true_robot_velocities()

        self.initial_robot_distance = 0.0


        # Manual control parameters
        self.manual_control = False
        self._last_toggle_time = 0.0
        self.il_buffer = []
        
        # Timing parameters
        
        self.humans_dt = HUMANS_DT
        self.max_episode_time = MAX_EPISODE_TIME

        
        
        # Physics parameters
        self.robot_radius = ROBOT_RADIUS
        
        # Initialize humans-related components
        self.human_parameters = get_standard_humans_parameters(self.n_humans+1)
        self.human_parameters = self.human_parameters.at[-1, 2].set(MAX_LIN_VEL_ROBOT) # or 0.25
        self.human_parameters = self.human_parameters.at[-1, 6].set(REPELLENT_FORCE) # or 0.25
        self.human_parameters = self.human_parameters.at[-1, 10].set(REPELLENT_WALL_FORCE) # or 0.25


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
            scenario1_nohumans.scenario1_nohumans: 14
        }

        #listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        #listener.start()
        
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
        
        # Load scenarios once for better performance
        self.scenarios = [
            scenario1_easy.scenario1_easy,
            scenario4.scenario4, # Corridoio
            scenario9.scenario9, # Scenario con ostacoli    
            scenario12.scenario12, # Scenario con ostacoli e robot
            scenario1_nohumans.scenario1_nohumans, # Scenario senza umani
        ]

        #self.scenarios = [scenario1_nh.scenario1_nh, scenario2_nh.scenario2_nh, scenario3_nh.scenario3_nh,
                        #   scenario4_nh.scenario4_nh, scenario5_nh.scenario5_nh, scenario6_nh.scenario6_nh,
                        #   scenario7_nh.scenario7_nh, scenario8_nh.scenario8_nh, scenario9_nh.scenario9_nh,
                        #   scenario10_nh.scenario10_nh, scenario11_nh.scenario11_nh, scenario12_nh.scenario12_nh]

        #self.scenarios = [scenario9.scenario9, scenario10.scenario10]
        
        # Initialize MuJoCo environment
        self._setup_mujoco()

        human_names = [f"human{i+1}" for i in range(self.n_humans)]
        self.collision_detector = CollisionDetector(self.model, self.data, robot_body_name="agent_body", human_body_names=human_names)
        self.robot_dt = ROBOT_DT
        
        # Setup viewer if needed
        # 
        self.viewer = None
        # if self.render_mode == "human":
        #     self._setup_viewer()
        
        # Initialize environment
        self.reset()

    def _keyboard_action(self) -> np.ndarray:
        """Handle keyboard actions."""
        lin, ang = 0.0, 0.0
        if FWD_KEY in key_pressed: lin += 1.0
        if BACK_KEY in key_pressed: lin -= 1.0
        if LEFT_KEY in key_pressed: ang += 1.0
        if RIGHT_KEY in key_pressed: ang -= 1.0
        return np.arra([np.clip(lin, 0.0, 1.0), np.clip(ang, -1.0, 1.0)], dtype=np.float32)
    
    def _apply_manual_action(self, action: np.ndarray, dt = ROBOT_DT):
        """Directly apply manual action to the robot using keyboard input."""
        lin_vel = float(action[0])*MAX_LIN_VEL_ROBOT
        ang_vel = float(action[1])  

        x,y, theta = self.data.qpos[:3] 
        if abs(ang_vel) > 0.001:
            R = lin_vel/ang_vel if ang_vel != 0 else 0.0
            x += R * (np.sin(theta + ang_vel * dt) - np.sin(theta))
            y -= R * (np.cos(theta) - np.cos(theta + ang_vel * dt))
            theta += ang_vel * dt
        else:
            x += lin_vel * np.cos(theta) * dt
            y += lin_vel * np.sin(theta) * dt   

        self.data.qpos[:3] = [x, y, theta]

    def _initialize_obstacles(self):
        """Initialize obstacle data efficiently."""
        self.obstacles = None
        self.all_obstacles = True # passa tutti gli ostacoli direttamente
        self.humans_state_numpy = np.zeros((self.n_humans+1, 6), dtype=np.float32)
        
        if self.all_obstacles:
            obstacles_data = self._get_all_obstacles()
            if obstacles_data is not None:
                self.obstacles = jnp.stack([obstacles_data] * (self.n_humans+1))
    
    def _reset_episode_counters(self):
        """Reset all episode-related counters."""
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
    
        
        # Rates (computed properties)
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
        """Initialize state tracking variables."""
        self.humans_goals = jnp.zeros((self.n_humans+1, 2, 2), dtype=jnp.float32)
        self.humans_current_goals = jnp.zeros((self.n_humans+1, 2), dtype=jnp.float32)
        self.robot_pos = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.robot_rot_matrix = np.eye(3, dtype=np.float32)
    
    def _initialize_observation_stacking(self):
        """Initialize observation stacking components."""
        # Initialize deque to store lidar observations with fixed length
        self.lidar_stack = deque(maxlen=self.n_stacking)
        
        # Initialize with zeros (will be filled during first reset)
        empty_lidar = np.zeros(self.num_rays, dtype=np.float32) # one lidar frame initialized as zeros
        for _ in range(self.n_stacking):
            self.lidar_stack.append(empty_lidar.copy()) # initialize stack with empty lidar readings
    
    def _setup_spaces(self):
        """Setup action and observation spaces with proper dtypes."""
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space now includes stacked lidar readings and stacked polar data
        # [stacked_lidar_readings, stacked_distance_and_angle]
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
        
        # Initialize human body IDs
        self.human_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            for i in range(self.n_humans)
        ], dtype=np.int32)

        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]


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
        self.lidar_stack.append(new_lidar_reading.copy())
    
    def _get_stacked_lidar_obs(self) -> np.ndarray:
        """Get stacked lidar observations as a flattened array."""
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

            total_episodes = self.episode_count 
            total_counted = self.success_count + self.collision_count + self.timeout_count
            
            # Calculate rates
            if total_episodes > 1:
                self.success_rate = self.success_count / total_episodes
                self.collision_rate = self.collision_count / total_episodes
                self.human_collision_rate = self.human_collision_count / total_episodes
                self.timeout_rate = self.timeout_count / total_episodes
                
                # Log results for evaluation
                if not self.training:
                    self._log_episode_result(total_episodes)

    
    def _log_episode_result(self, episode_num: int):
        """Log episode results during evaluation."""
        result_msg = (f"{self.last_episode_result.upper()}: "
                     f"Episode={episode_num} sr={self.success_rate:.2f}, "
                     f"cr={self.collision_rate:.2f}, hcr={self.human_collision_rate:.2f}, tr={self.timeout_rate:.2f}, ")
        #print(result_msg)
    
    def _load_random_scenario(self) -> Dict[str, float]:
        """Load a random scenario efficiently."""
        scenario_func = random.choice(self.scenarios)
        #print(f"Loading scenario: {scenario_func.__name__}")

        self.current_scenario_id = self.scenario_mapping.get(scenario_func, 0)
        return scenario_func()
    
    # def _set_robot_initial_state(self, scenario_data: Dict[str, float]): # don't need this function since the robot is now a human
    #     """Set robot initial position and orientation."""
    #     self.data.qpos[0] = scenario_data["mob_robot_startposx"]
    #     self.data.qpos[1] = scenario_data["mob_robot_startposy"] 
    #     self.data.qpos[2] = scenario_data["mob_robot_start_orientation"]
    #     self.data.qvel[:] = np.zeros_like(self.data.qvel)

    
    def _set_humans_initial_state(self, scenario_data: Dict[str, float]):
        """Set humans initial positions and goals efficiently."""
        # Initialize arrays for positions and goals
        human_positions = []
        human_goals = []
        
        # Process regular humans first (indices 0 to n_humans-1)
        for i in range(self.n_humans):
            # Human positions and goals
            human_pos = (scenario_data[f"human{i+1}x"], scenario_data[f"human{i+1}y"])
            human_goal = (scenario_data[f"targethuman{i+1}x"], scenario_data[f"targethuman{i+1}y"])
            orientation = scenario_data[f"start_orientation_human{i+1}"]
            
            human_positions.append(human_pos)
            human_goals.append(human_goal)
            
            # Set position in MuJoCo model
            human_id = self.human_body_ids[i]  # Correct index for humans
            self.model.body_pos[human_id, :2] = [human_pos[0], human_pos[1]]
            self.model.body_quat[human_id] = [
                np.cos(orientation / 2), 0.0, 0.0, np.sin(orientation / 2)
            ]
        
        # Add robot as the last "human" (index n_humans)
        robot_pos = (scenario_data["mob_robot_startposx"], scenario_data["mob_robot_startposy"])
        robot_goal = (scenario_data["target_robot_x"], scenario_data["target_robot_y"])
        robot_orientation = scenario_data["mob_robot_start_orientation"]
        
        human_positions.append(robot_pos)
        human_goals.append(robot_goal)

        self.target_pos[:2] = robot_goal  # Set target position for the 
        
        # Place the target sphere in MuJoCo
        target_sphere_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if target_sphere_id >= 0:
            self.model.geom_pos[target_sphere_id, :2] = [robot_goal[0], robot_goal[1]]
        else:
            logging.warning("Target sphere geom not found in MuJoCo model.")
        
        # Set robot position in MuJoCo model
        robot_id = self.mobile_robot_ID
        self.model.body_pos[robot_id, :2] = [robot_pos[0], robot_pos[1]]
        self.model.body_quat[robot_id] = [
            np.cos(robot_orientation / 2), 0.0, 0.0, np.sin(robot_orientation / 2)
        ]
        
        # Convert to proper JAX arrays
        # Assuming each "human" (including robot) has a start goal and an end goal
        # that they alternate between. For now, we'll set both goals to the target.
        goals_array = []
        current_goals = []
        
        for i, (pos, goal) in enumerate(zip(human_positions, human_goals)):
            # For this implementation, we'll use the target as both goals
            # You might want to modify this based on your specific scenario structure
            goals_array.append([goal, pos])  # Goal and return position
            current_goals.append(goal)  # Start by going to the main goal
        
        self.humans_goals = jnp.array(goals_array, dtype=jnp.float32)  # Shape: (n_humans+1, 2, 2)
        self.humans_current_goals = jnp.array(current_goals, dtype=jnp.float32)  # Shape: (n_humans+1, 2)
    
    def _initialize_humans_tracking(self):
        """Initialize humans state tracking arrays."""
        self.humans_state_numpy = np.zeros((self.n_humans+1, 6), dtype=np.float32)
       
        for i in range(self.n_humans):
            human_id = self.human_body_ids[i]

            self.humans_state_numpy[i, :2] = self.model.body_pos[human_id, :2]
            
            self.humans_state_numpy[i, 2:4] = 0.0 # Velocity (initially zero) - will be updated by HSFM
            
            # Orientation from quaternion
            quat = self.model.body_quat[human_id]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            
            # Convert quaternion to yaw angle (rotation around z-axis)
            # Formula: theta = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
            theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            self.humans_state_numpy[i, 4] = theta
            
            # Angular velocity (initially zero) - will be updated by HSFM
            self.humans_state_numpy[i, 5] = 0.0

        # Then for the robot 
        robot_id = self.mobile_robot_ID
        self.humans_state_numpy[self.n_humans, :2] = self.model.body_pos[robot_id, :2]
        self.humans_state_numpy[self.n_humans, 2:4] = 0.0  # Initial velocity
        quat = self.model.body_quat[robot_id]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.humans_state_numpy[self.n_humans, 4] = theta  # Orientation
        self.humans_state_numpy[self.n_humans, 5] = 0.0  # Initial angular velocity

    def _get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        # Get robot position and orientation
        """Get current observation with [goal info | distances | angles | lidar stack]."""
        #info = self._get_info()

        # Update robot position
        self.robot_pos[:2] = self.humans_state_numpy[-1, :2]

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

        robot_orientation = self.humans_state_numpy[-1, 4]

        relative_angle = target_angle - robot_orientation
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        self.relative_angle = relative_angle  


        # print lidar
        
        return distance_to_target, relative_angle, current_lidar
































    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        previous_obs = self._get_obs()  # Get previous observation before step

        current_step_duration = time.time() - self.step_time_measure
        self.total_step_taken += 1  
        #print(f"Step {self.total_step_taken} - Duration: {current_step_duration:.4f}s")
        info = self._get_info()

        self.average_step_duration = (self.average_step_duration * (self.total_step_taken - 1) + (time.time() - self.step_time_measure)) / self.total_step_taken
        if self.average_step_duration > 100:
            self.average_step_duration = 0.008

        #print(f"Step duration measurement: {current_step_duration:.4f}s, Average: {self.average_step_duration:.4f}s") 

        self.step_time_measure = time.time()  # Start measuring step duration
        self.episode_time = time.time() - self.episode_time_begin
        self.robot_action_counter += 1
        
   
        # self._apply_robot_action(action) # No need for robot action, the robot is a human now
        if TOGGLE_KEY in key_pressed and time.time() - self._last_toggle_time > 0.3:
            self.manual_control = not self.manual_control
            self._last_toggle_time = time.time()
            mode = 'MANUAL' if self.manual_control else 'HSFM'
            print(f"[Toggle] Switching to {mode} control mode.")

        if self.manual_control:
            action = self._keyboard_action()  # Get action from keyboard input
            self._apply_manual_action(action, dt=self.robot_dt)  # Apply manual action directly
            obs = self._get_obs()  # Get observation after manual action
            self.il_buffer.append((obs.copy(), action.copy()))  # Store observation and action in buffer

        else:
            # Update humans simulation

            humansstate = self.humans_state_numpy.copy()
            self._update_humans_simulation()
            next_humans_state = self.humans_state_numpy.copy()
        
        actual_angle = next_humans_state[-1, 4]  # Robot's current orientation
        prev_angle = humansstate[-1, 4]  # Robot's previous orientation


        diff_angle = next_humans_state[-1, 4] - humansstate[-1, 4]
        
        wrapped_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi  # Wrap angle to [-pi, pi]

        ang_vel = wrapped_angle / self.robot_dt


        actual_relative_angle = self.relative_angle

        #print(f"Relative angle: {actual_relative_angle:.2f} rad, Angular velocity: {ang_vel:.2f} rad/s, ") 

        lin_vel = np.linalg.norm(next_humans_state[-1, 0:2] - humansstate[-1, 0:2]) / self.robot_dt
        

        # Update MuJoCo simulation
        mujoco.mj_forward(self.model, self.data)


        #mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get info first
        info = self._get_info()
        info["expert_action"] = np.array([lin_vel, ang_vel], dtype=np.float32)  # Store expert action for IL
        # Get observations
        observations = self._get_obs(info)
        
        # Calculate reward and check termination
        reward, terminated, truncated = self._calculate_reward_and_termination(info, self.episode_time, observations, previous_obs)
       
        #diff_actions = self.get_true_robot_velocities()
        

        self.render()
        
        self.episode_return += reward
        info["episode_result"] = self.last_episode_result
        
        return observations, reward, terminated, truncated, info
    
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

        # Get lidar stack
        lidar_stack = self._get_stacked_lidar_obs()  # shape: (n_stacking * num_rays,)

        # Final obs: [goal info] + [distances] + [angles] + [lidar]
        observation = np.concatenate([
            stacked_distances,
            stacked_angles,
            lidar_stack,
        ]).astype(np.float32)

        # observation = np.concatenate([
        #     np.array([current_target_distance]),
        #     np.array([current_relative_angle]),
        #     lidar_stack,
        # ]).astype(np.float32)

        return observation

    
    def _update_polar_stack(self, polar_data: np.ndarray) -> np.ndarray:
        """Update the polar stack with new data and return the flattened stack."""
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
          
    
    def _update_humans_simulation(self):
        """Update humans simulation using HSFM, with optional manual control for the robot."""

        if not self.all_obstacles:
            found_obstacles = self._get_obstacles_from_human_positions(self.humans_state_numpy)
            self.obstacles = self._get_static_obstacles_formatted(found_obstacles)

        self._update_human_goals()
        n_substeps = int(self.robot_dt / self.humans_dt)

        if self.manual_control:
            # MANUAL mode → exclude robot from HSFM
            humans_state_jax = jnp.array(self.humans_state_numpy[:-1], dtype=jnp.float32)
            goals_jax        = self.humans_current_goals[:-1]
            params_jax       = self.human_parameters[:-1]
            obstacles_jax    = self.obstacles[:-1] if self.obstacles is not None else None

            # substep loop (only for real humans)
            for _ in range(n_substeps):
                humans_state_jax = step(
                    humans_state_jax,
                    goals_jax,
                    params_jax,
                    obstacles_jax,
                    self.humans_dt,
                )
                if jnp.isnan(humans_state_jax).any():
                    logging.warning("NaN values detected in human state")
                    return

            # update only human rows
            self.humans_state_numpy[:-1] = np.array(humans_state_jax, dtype=np.float32)

            # update robot row from MuJoCo
            robot_id = self.mobile_robot_ID
            self.humans_state_numpy[-1, :2] = self.model.body_pos[robot_id, :2]
            self.humans_state_numpy[-1, 4]  = self.data.qpos[2]
            # velocities can stay at 0

        else:
            # HSFM mode → include robot in simulation
            humans_state_jax = jnp.array(self.humans_state_numpy, dtype=jnp.float32)
            goals_jax        = self.humans_current_goals
            params_jax       = self.human_parameters
            obstacles_jax    = self.obstacles if self.obstacles is not None else None

            for _ in range(n_substeps):
                humans_state_jax = step(
                    humans_state_jax,
                    goals_jax,
                    params_jax,
                    obstacles_jax,
                    self.humans_dt,
                )
                if jnp.isnan(humans_state_jax).any():
                    logging.warning("NaN values detected in human state")
                    return

            # full copy back
            self.humans_state_numpy = np.array(humans_state_jax, dtype=np.float32)

        # always update MuJoCo positions
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

        # Then for the robot (last human in the array)
        robot_id = self.mobile_robot_ID
        robot_pos = self.humans_state_numpy[-1, :2]
        robot_orientation = self.humans_state_numpy[-1, 4]
        self.model.body_pos[robot_id, :] = [robot_pos[0], robot_pos[1], 0.4]    
        half_angle = robot_orientation / 2
        self.model.body_quat[robot_id] = [
            np.cos(half_angle), 0., 0., np.sin(half_angle)
        ]
        #mujoco.mj_forward(self.model, self.data)
    
    def _calculate_reward_and_termination(self, info: Dict[str, Any], episode_time: float, obs, previous_obs) -> Tuple[float, bool, bool]:
        reward = 0.0
        terminated = False
        truncated = False
        
        # Unpack obs
        current_lidar = obs[self.n_stacking*(2+self.num_rays-1):]  # current Lidar readings are the end of the observation array
        # Get distance to target and relative angle
        current_target_distance = obs[self.n_stacking-1]
        current_relative_angle = obs[self.n_stacking*2-1]


        previous_distance = previous_obs[self.n_stacking-1] 

        # current_target_distance = obs[0]  # First element is distance to target
        # current_relative_angle = obs[1]  # Second element is relative angle
        # current_lidar = obs[2:]  # Remaining elements are lidar readings

        # previous_distance = previous_obs[0]  # First element of previous observation is distance to target


        
        # Progress reward
        progress_reward = PROGRESS_REWARD_SCALE*(previous_distance - current_target_distance)
        reward += progress_reward
        previous_distance = current_target_distance
        
        # Angle penalty
        angle_reward = -0.01*abs(current_relative_angle)
        reward += angle_reward

        # Success condition
        if current_target_distance < DISTANCE_SUCCESS_THRESHOLD:
            self.last_episode_result = "success"
            terminated = True
            reward += 200
            return reward, terminated, truncated
        
        # Collision detection
        collision_detected, rew_lasers = self._lasers_reward(reward, current_lidar)
        reward += rew_lasers
        
        humans_collision_detected, _ = self.collision_detector.check_robot_human_collision_distance(
            collision_threshold=COLLISION_THRESHOLD
        )
        
        if collision_detected or humans_collision_detected:
            reward += -20 if humans_collision_detected else -10
            terminated = True
            #terminated = True
        
        if episode_time >= self.max_episode_time:  # Use actual elapsed time
            self.last_episode_result = "timeout"
            truncated = True

            # Enhanced info dict
        info.update({
            'episode_result': self.last_episode_result,
            'termination_reason': self.last_episode_result
        })
        
        return reward, terminated, truncated
    
    def _lasers_reward(self, base_reward: float, current_lidar) -> bool:
        """Check for collisions and apply proximity penalties."""
        collision_detected = False
        base_reward = 0
        for i in range(0, len(current_lidar)):
            reading = current_lidar[i]

            if reading < COLLISION_THRESHOLD and reading > self.robot_radius: #0.4
                # Close to obstacle - apply penalty
                base_reward -= 0.1
            elif reading <= self.robot_radius:
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
        if rendering_disable:
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
    
    def _setup_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
