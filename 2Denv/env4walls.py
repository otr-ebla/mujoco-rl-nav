import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
import math
import random
from typing import List, Tuple, Optional

class RobotNavigationEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode: Optional[str] = None, width: int = 800, height: int = 600):
        super().__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.wall_thickness = 20
        self.robot_radius = 15
        self.target_radius = 10
        self.lidar_range = 200
        self.lidar_rays = 360  # 360 degree coverage, 1 degree per ray
        self.max_steps = 1000
        
        # Action space: [linear_velocity, angular_velocity]
        # Linear velocity: 0 to 1, Angular velocity: -1 to 1
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.5]), 
            high=np.array([0.01, 0.5]), 
            dtype=np.float32
        )
        
        # Observation space: lidar readings (360 values)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.lidar_range, 
            shape=(self.lidar_rays,), 
            dtype=np.float32
        )
        
        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Environment state
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_angle = 0.0
        self.target_pos = np.array([0.0, 0.0])
        self.obstacles = []
        self.step_count = 0
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        
    def _generate_obstacles(self, num_obstacles: int = 5) -> List[List[Tuple[float, float]]]:
        """Generate random shaped obstacles as polygons"""
        obstacles = []
        
        for _ in range(num_obstacles):
            # Random center position (avoiding edges)
            center_x = random.uniform(100, self.width - 100)
            center_y = random.uniform(100, self.height - 100)
            
            # Random polygon with 3-6 sides
            num_sides = random.randint(3, 6)
            radius = random.uniform(30, 60)
            
            vertices = []
            for i in range(num_sides):
                angle = (2 * math.pi * i) / num_sides + random.uniform(-0.5, 0.5)
                r = radius * random.uniform(0.5, 1.0)
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                vertices.append((x, y))
            
            obstacles.append(vertices)
        
        return obstacles
    
    def _get_valid_position(self, radius: float) -> np.ndarray:
        """Get a valid position that doesn't collide with obstacles or walls"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(self.wall_thickness + radius, 
                             self.width - self.wall_thickness - radius)
            y = random.uniform(self.wall_thickness + radius, 
                             self.height - self.wall_thickness - radius)
            pos = np.array([x, y])
            
            # Check collision with obstacles
            if not self._check_collision_with_obstacles(pos, radius):
                return pos
        
        # Fallback to center if no valid position found
        return np.array([self.width / 2, self.height / 2])
    
    def _check_collision_with_obstacles(self, pos: np.ndarray, radius: float) -> bool:
        """Check if a circle collides with any obstacle"""
        for obstacle in self.obstacles:
            if self._point_in_polygon_with_radius(pos, obstacle, radius):
                return True
        return False
    
    def _point_in_polygon_with_radius(self, point: np.ndarray, polygon: List[Tuple[float, float]], radius: float) -> bool:
        """Check if a circle with given radius intersects with a polygon"""
        # Simple approach: check if circle center is close to polygon edges
        x, y = point
        n = len(polygon)
        
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            
            # Distance from point to line segment
            A = x - x1
            B = y - y1
            C = x2 - x1
            D = y2 - y1
            
            dot = A * C + B * D
            len_sq = C * C + D * D
            
            if len_sq == 0:
                distance = math.sqrt(A * A + B * B)
            else:
                param = dot / len_sq
                if param < 0:
                    xx, yy = x1, y1
                elif param > 1:
                    xx, yy = x2, y2
                else:
                    xx = x1 + param * C
                    yy = y1 + param * D
                
                dx = x - xx
                dy = y - yy
                distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= radius:
                return True
        
        return False
    
    def _cast_ray(self, start_pos: np.ndarray, angle: float) -> float:
        """Cast a ray and return the distance to the nearest obstacle or wall"""
        min_distance = self.lidar_range
        
        # Ray starting point and direction
        x0, y0 = start_pos
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Check collision with walls (inner edges)
        walls = [
            # Left wall (inner edge)
            (self.wall_thickness, self.wall_thickness, self.wall_thickness, self.height - self.wall_thickness),
            # Right wall (inner edge)
            (self.width - self.wall_thickness, self.wall_thickness, self.width - self.wall_thickness, self.height - self.wall_thickness),
            # Top wall (inner edge)
            (self.wall_thickness, self.wall_thickness, self.width - self.wall_thickness, self.wall_thickness),
            # Bottom wall (inner edge)
            (self.wall_thickness, self.height - self.wall_thickness, self.width - self.wall_thickness, self.height - self.wall_thickness)
        ]
        
        for wall in walls:
            x1, y1, x2, y2 = wall
            distance = self._line_intersection(x0, y0, dx, dy, x1, y1, x2, y2)
            if distance is not None and distance < min_distance:
                min_distance = distance
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            n = len(obstacle)
            for i in range(n):
                x1, y1 = obstacle[i]
                x2, y2 = obstacle[(i + 1) % n]
                
                distance = self._line_intersection(x0, y0, dx, dy, x1, y1, x2, y2)
                if distance is not None and distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def _line_intersection(self, x0: float, y0: float, dx: float, dy: float, 
                          x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate intersection between ray and line segment"""
        # Ray: (x0, y0) + t * (dx, dy)
        # Line segment: (x1, y1) to (x2, y2)
        
        # Line segment direction
        sx = x2 - x1
        sy = y2 - y1
        
        # Check if ray and line segment are parallel
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None
        
        # Calculate intersection parameters
        t = ((x1 - x0) * sy - (y1 - y0) * sx) / denom
        u = ((x1 - x0) * dy - (y1 - y0) * dx) / denom
        
        # Check if intersection is valid:
        # t > 0: intersection is in front of ray origin
        # 0 <= u <= 1: intersection is within line segment
        if t > 0 and 0 <= u <= 1:
            return t
        
        return None
    
    def _get_lidar_readings(self) -> np.ndarray:
        """Get 360-degree lidar readings"""
        readings = np.zeros(self.lidar_rays)
        
        for i in range(self.lidar_rays):
            # Convert to global angle
            ray_angle = self.robot_angle + (i * 2 * math.pi / self.lidar_rays)
            readings[i] = self._cast_ray(self.robot_pos, ray_angle)
        
        return readings.astype(np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Generate obstacles
        self.obstacles = self._generate_obstacles()
        
        # Reset robot position and angle
        self.robot_pos = self._get_valid_position(self.robot_radius)
        self.robot_angle = random.uniform(0, 2 * math.pi)
        
        # Reset target position
        self.target_pos = self._get_valid_position(self.target_radius)
        
        # Ensure target is not too close to robot
        while np.linalg.norm(self.target_pos - self.robot_pos) < 50:
            self.target_pos = self._get_valid_position(self.target_radius)
        
        self.step_count = 0
        
        observation = self._get_lidar_readings()
        info = {}
        
        return observation, info
    
    def step(self, action):
        self.step_count += 1
        
        # Store previous distance to target
        prev_distance = np.linalg.norm(self.target_pos - self.robot_pos)
        
        # Extract actions
        linear_vel = action[0] * 100  # Scale to pixels per frame
        angular_vel = action[1] * 0.1  # Scale to radians per frame
        
        # Update robot state
        self.robot_angle += angular_vel
        self.robot_angle = self.robot_angle % (2 * math.pi)
        
        # Calculate new position
        new_pos = self.robot_pos + linear_vel * np.array([
            math.cos(self.robot_angle), 
            math.sin(self.robot_angle)
        ])
        
        # Check for collisions with walls
        wall_collision = False
        if not (self.wall_thickness + self.robot_radius <= new_pos[0] <= 
                self.width - self.wall_thickness - self.robot_radius and
                self.wall_thickness + self.robot_radius <= new_pos[1] <= 
                self.height - self.wall_thickness - self.robot_radius):
            wall_collision = True
        
        # Check for collisions with obstacles
        obstacle_collision = self._check_collision_with_obstacles(new_pos, self.robot_radius)
        
        # Update position if no collisions
        if not wall_collision and not obstacle_collision:
            self.robot_pos = new_pos
        
        # Get lidar readings
        observation = self._get_lidar_readings()
        min_distance = np.min(observation)
        
        # Calculate new distance to target
        distance_to_target = np.linalg.norm(self.target_pos - self.robot_pos)
        
        # Check termination conditions
        target_reached = distance_to_target < (self.robot_radius + self.target_radius)
        collision = wall_collision or obstacle_collision
        terminated = target_reached or collision
        truncated = self.step_count >= self.max_steps
        
        # REWARD SHAPING
        reward = 0.0
        
        # 1. Large positive reward for reaching target
        if target_reached:
            reward += 100.0
        
        # 2. Large negative reward for collision
        if collision:
            reward -= 50.0
        
        # 3. Progress reward (positive when getting closer to target)
        progress = (prev_distance - distance_to_target) * 0.1  # Scale factor
        reward += progress
        
        # 4. Small penalty for each time step (encourage efficiency)
        reward -= 0.1
        
        # 5. Penalty for getting too close to obstacles
        if min_distance < 20:
            reward -= (20 - min_distance) * 0.05
        
        # 6. Bonus for facing the target
        target_direction = np.arctan2(self.target_pos[1] - self.robot_pos[1],
                                    self.target_pos[0] - self.robot_pos[0])
        angle_diff = abs((self.robot_angle - target_direction + math.pi) % (2 * math.pi) - math.pi)
        facing_reward = 0.1 * (math.pi - angle_diff) / math.pi  # Max 0.1 when facing directly
        reward += facing_reward
        
        info = {
            'distance_to_target': distance_to_target,
            'min_lidar_distance': min_distance,
            'collision': collision,
            'target_reached': target_reached,
            'progress': progress,
            'facing_reward': facing_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Robot Navigation Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw walls
        pygame.draw.rect(self.screen, self.BLACK, (0, 0, self.width, self.wall_thickness))
        pygame.draw.rect(self.screen, self.BLACK, (0, self.height - self.wall_thickness, self.width, self.wall_thickness))
        pygame.draw.rect(self.screen, self.BLACK, (0, 0, self.wall_thickness, self.height))
        pygame.draw.rect(self.screen, self.BLACK, (self.width - self.wall_thickness, 0, self.wall_thickness, self.height))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.polygon(self.screen, self.GRAY, obstacle)
        
        # Draw target
        pygame.draw.circle(self.screen, self.GREEN, self.target_pos.astype(int), self.target_radius)
        
        # Draw robot
        pygame.draw.circle(self.screen, self.BLUE, self.robot_pos.astype(int), self.robot_radius)
        
        # Draw robot direction
        direction_end = self.robot_pos + 25 * np.array([math.cos(self.robot_angle), math.sin(self.robot_angle)])
        pygame.draw.line(self.screen, self.RED, self.robot_pos, direction_end, 3)
        
        # Draw lidar rays (optional, for visualization)
        if hasattr(self, '_show_lidar') and self._show_lidar:
            # Only update lidar visualization every few frames for performance
            if not hasattr(self, '_lidar_frame_counter'):
                self._lidar_frame_counter = 0
                self._cached_lidar_readings = None
            
            self._lidar_frame_counter += 1
            
            # Update lidar readings every 3 frames instead of every frame
            if self._lidar_frame_counter % 3 == 0 or self._cached_lidar_readings is None:
                self._cached_lidar_readings = self._get_lidar_readings()
            
            # Draw fewer rays for better performance (every 15th ray instead of 10th)
            for i in range(0, self.lidar_rays, 15):
                angle = self.robot_angle + (i * 2 * math.pi / self.lidar_rays)
                end_pos = self.robot_pos + self._cached_lidar_readings[i] * np.array([math.cos(angle), math.sin(angle)])
                pygame.draw.line(self.screen, self.YELLOW, self.robot_pos, end_pos, 1)
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
    
    def _render_rgb_array(self):
        # This would return a numpy array of the rendered screen
        # For now, just call human rendering
        self._render_human()
        return np.array(pygame.surfarray.array3d(self.screen))
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            
    def toggle_lidar_visualization(self):
        """Toggle lidar ray visualization"""
        self._show_lidar = not getattr(self, '_show_lidar', False)
        
    def get_robot_state(self):
        """Get current robot state for external control"""
        return {
            'position': self.robot_pos.copy(),
            'angle': self.robot_angle,
            'target_position': self.target_pos.copy(),
            'distance_to_target': np.linalg.norm(self.target_pos - self.robot_pos)
        }
    
    def set_robot_state(self, position: np.ndarray, angle: float):
        """Set robot state for external control"""
        # Check if position is valid
        if (self.wall_thickness + self.robot_radius <= position[0] <= 
            self.width - self.wall_thickness - self.robot_radius and
            self.wall_thickness + self.robot_radius <= position[1] <= 
            self.height - self.wall_thickness - self.robot_radius):
            
            if not self._check_collision_with_obstacles(position, self.robot_radius):
                self.robot_pos = position
                self.robot_angle = angle % (2 * math.pi)
                return True
        return False


# Register the environment
gym.register(
    id='RobotNavigation-v0',
    entry_point='__main__:RobotNavigationEnv',
    max_episode_steps=1000,
)