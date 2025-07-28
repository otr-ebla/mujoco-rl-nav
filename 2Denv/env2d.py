import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
import math
from typing import Tuple, Dict, List, Optional, Any
from collections import deque
import os
import time

# Constants
ROBOT_RADIUS = 10  # pixels
HUMAN_RADIUS = 8   # pixels
LIDAR_MAX_DISTANCE = 200  # pixels
DISTANCE_SUCCESS_THRESHOLD = 20  # pixels
MAX_EPISODE_TIME = 120  # seconds
ROBOT_DT = 0.25  # control timestep in seconds
MAX_LIN_VEL_ROBOT = 50  # pixels/second
MAX_ANG_VEL_ROBOT = 2.0  # rad/second
PROGRESS_REWARD_SCALE = 0.03
N_STACKING = 10  # Default stacking size for observations

class HumanAware2DEnv(gym.Env):
    """2D human-aware navigation environment with Pygame visualization."""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, 
                 num_rays: int = 36,
                 n_humans: int = 5,
                 render_mode: Optional[str] = None,
                 n_stacking: int = N_STACKING,
                 enable_stacking: bool = True,
                 screen_size: Tuple[int, int] = (800, 600)):
        
        # Core parameters
        self.num_rays = num_rays
        self.n_humans = n_humans
        self.render_mode = render_mode
        self.n_stacking = n_stacking
        self.enable_stacking = enable_stacking
        self.screen_size = screen_size
        self.screen = None
        self.clock = None
        
        # World boundaries
        self.world_width = screen_size[0]
        self.world_height = screen_size[1]
        
        # Initialize state variables
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.humans_pos = np.zeros((n_humans, 2), dtype=np.float32)
        self.humans_goals = np.zeros((n_humans, 2, 2), dtype=np.float32)  # Each human has 2 goals
        self.humans_current_goals = np.zeros((n_humans, 2), dtype=np.float32)
        self.humans_vel = np.zeros((n_humans, 2), dtype=np.float32)
        
        # Obstacles (simple rectangles for this 2D version)
        self.obstacles = [
            {'pos': [200, 150], 'size': [20, 200]},
            {'pos': [400, 300], 'size': [200, 20]},
            {'pos': [600, 450], 'size': [20, 200]},
        ]
        
        # Episode tracking
        self._reset_episode_counters()
        
        # Initialize observation stacking
        self._initialize_observation_stacking()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Colors
        self.colors = {
            'robot': (0, 0, 255),
            'target': (0, 255, 0),
            'humans': (255, 0, 0),
            'obstacles': (100, 100, 100),
            'lidar': (255, 255, 0, 100),
            'background': (255, 255, 255)
        }
        
        # Initialize scenario system
        self._setup_scenarios()
        
    def _reset_episode_counters(self):
        """Reset all episode-related counters."""
        self.current_step = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.episode_return = 0
        self.previous_distance = float('inf')
        self.last_episode_result = None
        self.episode_start_time = 0
        self.episode_time = 0.0
        
        # For observation stacking
        if self.enable_stacking:
            self.polar_stack = deque([np.zeros(2, dtype=np.float32) for _ in range(self.n_stacking)], 
                                   maxlen=self.n_stacking)
    
    def _initialize_observation_stacking(self):
        """Initialize observation stacking components."""
        self.lidar_stack = deque(maxlen=self.n_stacking)
        empty_lidar = np.zeros(self.num_rays, dtype=np.float32)
        for _ in range(self.n_stacking):
            self.lidar_stack.append(empty_lidar.copy())
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [linear_velocity (0-1), angular_velocity (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space depends on stacking
        if self.enable_stacking:
            # Stacked observations: [stacked_lidar, stacked_polar]
            stacked_lidar_size = self.num_rays * self.n_stacking
            stacked_polar_size = 2 * self.n_stacking  # distance and angle
            total_obs_size = stacked_lidar_size + stacked_polar_size
            
            obs_low = np.concatenate([
                np.zeros(stacked_lidar_size, dtype=np.float32),
                np.concatenate([
                    np.zeros(self.n_stacking, dtype=np.float32),  # distances
                    np.full(self.n_stacking, -np.pi, dtype=np.float32)  # angles
                    ])
                ])
            
            obs_high = np.concatenate([
                np.full(stacked_lidar_size, LIDAR_MAX_DISTANCE, dtype=np.float32),
                np.concatenate([
                    np.full(self.n_stacking, np.sqrt(self.world_width**2 + self.world_height**2)),  # max possible distance
                    np.full(self.n_stacking, np.pi, dtype=np.float32)  # angles
                    ])
                ])
        else:
            # Non-stacked: [lidar, distance, angle]
            total_obs_size = self.num_rays + 2
            obs_low = np.concatenate([
                np.zeros(self.num_rays, dtype=np.float32),
                np.array([0.0, -np.pi], dtype=np.float32)
            ])
            obs_high = np.concatenate([
                np.full(self.num_rays, LIDAR_MAX_DISTANCE, dtype=np.float32),
                np.array([np.sqrt(self.world_width**2 + self.world_height**2), np.pi], dtype=np.float32)
            ])
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_obs_size,),
            dtype=np.float32
        )
    
    def _setup_scenarios(self):
        """Setup navigation scenarios."""
        self.scenarios = [
            self._create_scenario1,
            self._create_scenario2,
            self._create_scenario3
        ]
        
        self.scenario_successes = {i: 0 for i in range(len(self.scenarios))}
        self.scenario_attempts = {i: 1 for i in range(len(self.scenarios))}  # Avoid division by zero
        self.current_scenario_id = None
    
    def _create_scenario1(self):
        """Simple scenario with robot on left, target on right."""
        return {
            'robot_start': [100, 300],
            'robot_angle': 0,
            'target': [700, 300],
            'humans': [
                {'start': [300, 200], 'goals': [[300, 200], [300, 400]]},
                {'start': [500, 400], 'goals': [[500, 400], [500, 200]]}
            ]
        }
    
    def _create_scenario2(self):
        """Scenario with obstacles in the middle."""
        return {
            'robot_start': [100, 300],
            'robot_angle': 0,
            'target': [700, 300],
            'humans': [
                {'start': [400, 200], 'goals': [[400, 200], [400, 400]]},
                {'start': [600, 300], 'goals': [[600, 300], [600, 100]]}
            ]
        }
    
    def _create_scenario3(self):
        """More complex scenario with multiple humans."""
        return {
            'robot_start': [100, 100],
            'robot_angle': np.pi/4,
            'target': [700, 500],
            'humans': [
                {'start': [300, 300], 'goals': [[300, 300], [300, 100]]},
                {'start': [500, 200], 'goals': [[500, 200], [500, 400]]},
                {'start': [400, 400], 'goals': [[400, 400], [400, 200]]}
            ]
        }
    
    def _load_random_scenario(self):
        """Load a random scenario with probability based on success rates."""
        if self.render_mode is not None:  # During evaluation, choose randomly
            scenario_func = random.choice(self.scenarios)
            self.current_scenario_id = self.scenarios.index(scenario_func)
            return scenario_func()
        
        # During training, weight by failure rate
        weights = []
        for i, func in enumerate(self.scenarios):
            success_rate = self.scenario_successes[i] / self.scenario_attempts[i]
            weight = 1.0 - success_rate
            weights.append(max(weight, 0.01))  # Avoid zero probability
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        scenario_func = random.choices(self.scenarios, weights=weights, k=1)[0]
        self.current_scenario_id = self.scenarios.index(scenario_func)
        return scenario_func()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self._reset_episode_counters()
        self.episode_start_time = time.time()
        
        # Load scenario
        scenario = self._load_random_scenario()
        
        # Set robot initial state
        self.robot_pos = np.array(scenario['robot_start'], dtype=np.float32)
        self.robot_angle = scenario['robot_angle']
        
        # Set target position
        self.target_pos = np.array(scenario['target'], dtype=np.float32)
        self.previous_distance = np.linalg.norm(self.target_pos - self.robot_pos)
        
        # Set humans initial state
        self.humans_pos = np.zeros((self.n_humans, 2), dtype=np.float32)
        self.humans_goals = np.zeros((self.n_humans, 2, 2), dtype=np.float32)
        self.humans_current_goals = np.zeros((self.n_humans, 2), dtype=np.float32)
        
        for i, human in enumerate(scenario['humans'][:self.n_humans]):
            self.humans_pos[i] = np.array(human['start'], dtype=np.float32)
            self.humans_goals[i] = np.array(human['goals'], dtype=np.float32)
            self.humans_current_goals[i] = self.humans_goals[i, 0]
        
        # Initialize observation stack
        if self.enable_stacking:
            self._reset_observation_stack()
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """Execute one environment step."""
        # Apply robot action
        self._apply_robot_action(action)
        
        # Update humans
        self._update_humans()
        
        # Update time
        self.current_step += 1
        self.episode_time = time.time() - self.episode_start_time
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        # Calculate reward and termination
        reward, terminated, truncated = self._calculate_reward_and_termination(info)
        self.episode_return += reward
        
        # Rendering
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_robot_action(self, action):
        """Apply robot action using kinematic model."""
        # Normalize action
        lin_vel = float(action[0]) * MAX_LIN_VEL_ROBOT  # 0 to max
        ang_vel = float(action[1]) * MAX_ANG_VEL_ROBOT  # -max to max
        
        # Update robot angle
        self.robot_angle += ang_vel * ROBOT_DT
        self.robot_angle = (self.robot_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize
        
        # Update robot position
        direction = np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        self.robot_pos += direction * lin_vel * ROBOT_DT
        
        # Clip to world boundaries
        self.robot_pos[0] = np.clip(self.robot_pos[0], ROBOT_RADIUS, self.world_width - ROBOT_RADIUS)
        self.robot_pos[1] = np.clip(self.robot_pos[1], ROBOT_RADIUS, self.world_height - ROBOT_RADIUS)
    
    def _update_humans(self):
        """Update human positions using simple goal-seeking behavior."""
        for i in range(self.n_humans):
            # Simple goal-seeking behavior
            direction = self.humans_current_goals[i] - self.humans_pos[i]
            distance = np.linalg.norm(direction)
            
            if distance < 5:  # Reach threshold
                # Switch goal
                if np.allclose(self.humans_current_goals[i], self.humans_goals[i, 0]):
                    self.humans_current_goals[i] = self.humans_goals[i, 1]
                else:
                    self.humans_current_goals[i] = self.humans_goals[i, 0]
                direction = self.humans_current_goals[i] - self.humans_pos[i]
                distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                speed = 20  # pixels/second
                self.humans_pos[i] += direction * speed * ROBOT_DT
            
            # Clip to world boundaries
            self.humans_pos[i][0] = np.clip(self.humans_pos[i][0], HUMAN_RADIUS, self.world_width - HUMAN_RADIUS)
            self.humans_pos[i][1] = np.clip(self.humans_pos[i][1], HUMAN_RADIUS, self.world_height - HUMAN_RADIUS)
    
    def _get_obs(self):
        """Get current observation."""
        # Get lidar readings
        lidar_readings = self._get_lidar_readings()
        
        # Update lidar stack
        if self.enable_stacking:
            self._update_lidar_stack(lidar_readings)
            lidar_obs = self._get_stacked_lidar_obs()
        else:
            lidar_obs = lidar_readings
        
        # Get target info
        distance_to_target = np.linalg.norm(self.target_pos - self.robot_pos)
        target_direction = self.target_pos - self.robot_pos
        target_angle = np.arctan2(target_direction[1], target_direction[0])
        relative_angle = target_angle - self.robot_angle
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize
        
        current_polar = np.array([distance_to_target, relative_angle], dtype=np.float32)
        
        if self.enable_stacking:
            polar_obs = self._update_polar_stack(current_polar)
        else:
            polar_obs = current_polar
        
        # Combine observations
        observation = np.concatenate([lidar_obs, polar_obs]).astype(np.float32)
        return observation
    
    def _get_lidar_readings(self):
        """Simulate 2D LIDAR sensor."""
        readings = np.zeros(self.num_rays, dtype=np.float32)
        angle_step = 2 * np.pi / self.num_rays
        
        for i in range(self.num_rays):
            angle = self.robot_angle + i * angle_step
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Initialize to max distance
            distance = LIDAR_MAX_DISTANCE
            
            # Check for collisions with walls
            wall_dist = self._raycast_walls(self.robot_pos, direction)
            distance = min(distance, wall_dist)
            
            # Check for collisions with obstacles
            for obstacle in self.obstacles:
                obs_dist = self._raycast_rectangle(
                    self.robot_pos, direction,
                    obstacle['pos'], obstacle['size']
                )
                distance = min(distance, obs_dist)
            
            # Check for collisions with humans
            for human_pos in self.humans_pos:
                human_dist = self._raycast_circle(
                    self.robot_pos, direction,
                    human_pos, HUMAN_RADIUS
                )
                distance = min(distance, human_dist)
            
            readings[i] = distance
        
        return readings
    
    def _raycast_walls(self, origin, direction):
        """Calculate distance to world boundaries."""
        # Calculate intersections with each wall
        t_values = []
        
        # Left wall (x=0)
        if direction[0] < 0:
            t = -origin[0] / direction[0]
            y = origin[1] + t * direction[1]
            if 0 <= y <= self.world_height:
                t_values.append(t)
        
        # Right wall (x=world_width)
        if direction[0] > 0:
            t = (self.world_width - origin[0]) / direction[0]
            y = origin[1] + t * direction[1]
            if 0 <= y <= self.world_height:
                t_values.append(t)
        
        # Top wall (y=0)
        if direction[1] < 0:
            t = -origin[1] / direction[1]
            x = origin[0] + t * direction[0]
            if 0 <= x <= self.world_width:
                t_values.append(t)
        
        # Bottom wall (y=world_height)
        if direction[1] > 0:
            t = (self.world_height - origin[1]) / direction[1]
            x = origin[0] + t * direction[0]
            if 0 <= x <= self.world_width:
                t_values.append(t)
        
        return min(t_values) if t_values else LIDAR_MAX_DISTANCE
    
    def _raycast_rectangle(self, origin, direction, rect_pos, rect_size):
        """Calculate distance to rectangle obstacle."""
        # Rectangle bounds
        rect_min = np.array(rect_pos)
        rect_max = rect_min + np.array(rect_size)
        
        # Calculate intersections
        t1 = (rect_min[0] - origin[0]) / direction[0] if direction[0] != 0 else float('inf')
        t2 = (rect_max[0] - origin[0]) / direction[0] if direction[0] != 0 else float('inf')
        t3 = (rect_min[1] - origin[1]) / direction[1] if direction[1] != 0 else float('inf')
        t4 = (rect_max[1] - origin[1]) / direction[1] if direction[1] != 0 else float('inf')
        
        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))
        
        if tmax < 0 or tmin > tmax:
            return LIDAR_MAX_DISTANCE
        
        t = tmin if tmin > 0 else tmax
        return t if t > 0 else LIDAR_MAX_DISTANCE
    
    def _raycast_circle(self, origin, direction, center, radius):
        """Calculate distance to circular obstacle (human)."""
        # Vector from origin to circle center
        oc = center - origin
        
        # Projection of oc onto direction
        proj = np.dot(oc, direction)
        
        # Closest point on ray to circle center
        closest = origin + proj * direction
        
        # Distance from closest point to circle center
        dist = np.linalg.norm(closest - center)
        
        if dist > radius:
            return LIDAR_MAX_DISTANCE
        
        # Calculate intersection points
        a = np.dot(direction, direction)
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return LIDAR_MAX_DISTANCE
        
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return LIDAR_MAX_DISTANCE
    
    def _update_lidar_stack(self, new_lidar_reading):
        """Update the lidar stack with new reading."""
        self.lidar_stack.append(new_lidar_reading.copy())
    
    def _get_stacked_lidar_obs(self):
        """Get stacked lidar observations as flattened array."""
        return np.concatenate(list(self.lidar_stack))
    
    def _update_polar_stack(self, polar_data):
        """Update the polar stack with new data."""
        self.polar_stack.append(polar_data.copy())
        return np.concatenate(list(self.polar_stack))
    
    def _calculate_reward_and_termination(self, info):
        """Calculate reward and check termination conditions."""
        reward = 0.0
        terminated = False
        truncated = False
        
        distance_to_target = info['distance_to_target']
        
        # Progress reward
        progress_reward = PROGRESS_REWARD_SCALE * (self.previous_distance - distance_to_target)
        reward += progress_reward
        self.previous_distance = distance_to_target
        
        # Angle penalty
        angle_penalty = -0.01 * abs(info['relative_angle'])
        reward += angle_penalty
        
        # Check for success
        if distance_to_target < DISTANCE_SUCCESS_THRESHOLD:
            reward += 200
            terminated = True
            self.last_episode_result = 'success'
            if self.current_scenario_id is not None:
                self.scenario_successes[self.current_scenario_id] += 1
        
        # Check for collisions
        collision_penalty, collision_detected = self._check_collisions()
        reward += collision_penalty
        
        if collision_detected:
            terminated = True
            self.last_episode_result = 'collision'
        
        # Check for timeout
        if self.episode_time > MAX_EPISODE_TIME:
            truncated = True
            self.last_episode_result = 'timeout'
        
        # Update scenario attempts
        if self.current_scenario_id is not None and terminated or truncated:
            self.scenario_attempts[self.current_scenario_id] += 1
        
        return reward, terminated, truncated
    
    def _check_collisions(self):
        """Check for collisions with obstacles or humans."""
        penalty = 0.0
        collision_detected = False
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            obs_min = np.array(obstacle['pos'])
            obs_max = obs_min + np.array(obstacle['size'])
            
            # Simple AABB collision check
            if (obs_min[0] < self.robot_pos[0] + ROBOT_RADIUS and
                obs_max[0] > self.robot_pos[0] - ROBOT_RADIUS and
                obs_min[1] < self.robot_pos[1] + ROBOT_RADIUS and
                obs_max[1] > self.robot_pos[1] - ROBOT_RADIUS):
                penalty -= 100
                collision_detected = True
        
        # Check human collisions
        for human_pos in self.humans_pos:
            dist = np.linalg.norm(human_pos - self.robot_pos)
            if dist < ROBOT_RADIUS + HUMAN_RADIUS:
                penalty -= 100
                collision_detected = True
        
        # Add proximity penalty based on lidar
        lidar_readings = self._get_lidar_readings()
        close_reading_threshold = ROBOT_RADIUS * 3
        
        for reading in lidar_readings:
            if reading < close_reading_threshold:
                penalty -= 0.1 * (close_reading_threshold - reading)
        
        return penalty, collision_detected
    
    def _get_info(self):
        """Get environment info dictionary."""
        target_direction = self.target_pos - self.robot_pos
        distance_to_target = np.linalg.norm(target_direction)
        target_angle = np.arctan2(target_direction[1], target_direction[0])
        relative_angle = target_angle - self.robot_angle
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize
        
        return {
            'distance_to_target': distance_to_target,
            'relative_angle': relative_angle,
            'robot_position': self.robot_pos.copy(),
            'target_position': self.target_pos.copy(),
            'episode_time': self.episode_time,
            'episode_result': self.last_episode_result,
            'scenario_id': self.current_scenario_id
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self._render_frame()
        elif self.render_mode == 'rgb_array':
            return self._render_frame(return_rgb_array=True)
    
    def _render_frame(self, return_rgb_array=False):
        """Render a single frame."""
        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Human-Aware 2D Navigation")
            self.clock = pygame.time.Clock()
        
        if self.render_mode == 'human' and self.screen is None:
            return
        
        # Create a surface if we're returning an rgb array
        if return_rgb_array or self.screen is None:
            surface = pygame.Surface(self.screen_size)
        else:
            surface = self.screen
        
        # Clear screen
        surface.fill(self.colors['background'])
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(
                surface, self.colors['obstacles'],
                (obstacle['pos'][0], obstacle['pos'][1], obstacle['size'][0], obstacle['size'][1])
            )
        
        # Draw target
        pygame.draw.circle(
            surface, self.colors['target'],
            (int(self.target_pos[0]), int(self.target_pos[1])),
            ROBOT_RADIUS
        )
        
        # Draw humans
        for human_pos in self.humans_pos:
            pygame.draw.circle(
                surface, self.colors['humans'],
                (int(human_pos[0]), int(human_pos[1])),
                HUMAN_RADIUS
            )
        
        # Draw robot
        pygame.draw.circle(
            surface, self.colors['robot'],
            (int(self.robot_pos[0]), int(self.robot_pos[1])),
            ROBOT_RADIUS
        )
        
        # Draw robot orientation
        end_pos = (
            self.robot_pos[0] + np.cos(self.robot_angle) * ROBOT_RADIUS * 1.5,
            self.robot_pos[1] + np.sin(self.robot_angle) * ROBOT_RADIUS * 1.5
        )
        pygame.draw.line(
            surface, (0, 0, 0),
            (int(self.robot_pos[0]), int(self.robot_pos[1])),
            (int(end_pos[0]), int(end_pos[1])),
            2
        )
        
        # Draw LIDAR rays (only for visualization)
        lidar_readings = self._get_lidar_readings()
        angle_step = 2 * np.pi / self.num_rays
        
        for i in range(self.num_rays):
            angle = self.robot_angle + i * angle_step
            end_x = self.robot_pos[0] + np.cos(angle) * lidar_readings[i]
            end_y = self.robot_pos[1] + np.sin(angle) * lidar_readings[i]
            
            pygame.draw.line(
                surface, self.colors['lidar'],
                (int(self.robot_pos[0]), int(self.robot_pos[1])),
                (int(end_x), int(end_y)),
                1
            )
        
        if return_rgb_array:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
            )
        
        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        """Close the environment and any open windows."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None