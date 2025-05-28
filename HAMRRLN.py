import numpy as np
import gymnasium as gym 
import mujoco
import mujoco.viewer
from mobilerobotRL import mobilerobotRL
import os
import xml.etree.ElementTree as ET
import time
import random
from scenarios.scenario1 import scenario1
from scenarios.scenario2 import scenario2   
from scenarios.scenario3 import scenario3
from scenarios.scenario4 import scenario4
from scenarios.scenario5 import scenario5
from scenarios.scenario6 import scenario6
from scenarios.scenario7 import scenario7
from scenarios.scenario8 import scenario8

import jax.numpy as jnp
from JHSFM.jhsfm.hsfm import step
from JHSFM.jhsfm.utils import get_standard_humans_parameters
from grid_decomp.labeled_grid import GridCell_operations
from torch.utils.tensorboard import SummaryWriter

class hamrrln(mobilerobotRL):
    def __init__(self, num_rays=108, model_path="assets/world.xml", training=True, n_humans = 5, log_dir="TENSORBOARD/", render_mode=None):
        #super().__init__(num_rays=num_rays, training=training_mode, model_path=model_path, )
        
        self.robot_dt = 0.25
        self.humans_dt = 0.01
        self.humans_state = None
        self.n_humans = n_humans
        self.human_parameters = get_standard_humans_parameters(self.n_humans)    
        self.humans_goals = jnp.zeros((self.n_humans, 2), dtype=float)
        self.grid_cell_op = GridCell_operations(cell_size=4, world_size=320)
        self.obstacles = None
        self.all_obstacles = True
        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=float)

        if self.all_obstacles:
                self.obstacles = jnp.stack([self.get_all_obstacles()] * self.n_humans)
                #print(f"All obstacles loaded: {self.obstacles.shape}")

        self.robot_radius = 0.3

        self.render_mode = render_mode if not training else None
        self.training = training
        self.num_rays = num_rays
        self.max_episode_time = 60
        self.current_step = 0
        self.previous_distance = 100
        self.episode_return = 0
        self.mean_episode_return = 0
        self.episode_counter = 0
        self.success_counter = 0
        self.collision_counter = 0
        self.timeout_counter = 0
        self.last_episode_result = None
        self.episode_time_length = 0
        self.episode_time_begin = 0
        self.lidar_readings = None  
        self.success_rate = 0
        self.collision_rate = 0
        self.timeout_rate = 0
        self.robot_relative_azimuth = 0
        self.model_path = model_path
        self.training_mode = training

        self.episode_time_begin = 0
        self.time_of_the_episode = 0

        self.humans_start_positions = jnp.zeros((self.n_humans, 2), dtype=float)    

        # robot_pos, target_pos, robot_rot_matrix, lidar_readings
        self.robot_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.robot_rot_matrix = np.eye(3)
        self.lidar_readings = np.zeros(self.num_rays)

        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0

        # Mobile Robot action space
        self.action_space = gym.spaces.Box(
            low = np.array([0, -1.0]),
            high = np.array([1.0, 1.0]),
            shape = (2, ),
            dtype = np.float32
        )

        # Mobile Robot observation space
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0]*num_rays+[0.0, -np.pi]),
            high = np.array([30.0]*num_rays+[0.0, np.pi]),
            shape = (num_rays + 2, ),
            dtype = np.float32
        )


        self.xml_model = self.load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model) 
        self.data = mujoco.MjData(self.model)

        self.mobile_robot_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]

        self.viewer = None
        if self.render_mode == "human" and not self.training:
            self._setup_viewer()

        self.reset()

            


    def reset(self, seed=None):
        self.episode_time_length = time.time() - self.episode_time_begin
        if self.current_step > 0:  # Only set this for non-first episodes
            self.last_episode_info = {"episode_time_length": self.episode_time_length}


        # Update episode statistics
        if self.last_episode_result == "success":
            self.success_count += 1
        elif self.last_episode_result == "collision":
            self.collision_count += 1
        elif self.last_episode_result == "timeout":
            self.timeout_count += 1
        epi_count = self.episode_count - 1       
        if self.episode_count > 1:
            
            self.success_rate = self.success_count / (epi_count)
            self.collision_rate = self.collision_count / (epi_count)
            self.timeout_rate = self.timeout_count / (epi_count)

        if self.last_episode_result == "success" and not self.training:
            print(f"SUCCESS: Eval_episode = {epi_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")
        elif self.last_episode_result == "collision" and self.training_mode == False:
            print(f"COLLISION: Eval_episode = {epi_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")
        elif self.last_episode_result == "timeout" and self.training_mode == False:
            print(f"TIMEOUT: Eval_episode = {epi_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")

        self.episode_count += 1
        self.last_episode_result = None
        self.episode_time_begin = time.time()
        
        super().reset(seed=seed)
        
        # Reset step counter and return
        self.current_step = 0
        self.episode_return = 0
        self.previous_distance = 200  # Reset distance tracking
        self.stuck_counter = 0  # Reset stuck counter

        self.time_of_the_episode = 0
        self.episode_time_begin = time.time()


        # Reset the environment
        # Choose a random scenario
        # Generate a random integer between 1 and 8
        scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8]
        random_scenario = random.randint(1, 8)
        data_scenario = scenarios[random_scenario - 1]()

        mob_robot_startposx = data_scenario["mob_robot_startposx"]  
        mob_robot_startposy = data_scenario["mob_robot_startposy"]
        mob_robot_start_orientation = data_scenario["mob_robot_start_orientation"]  
        self.data.qpos[0] = mob_robot_startposx
        self.data.qpos[1] = mob_robot_startposy
        self.data.qpos[2] = mob_robot_start_orientation

        target_robot_x = data_scenario["target_robot_x"]    
        target_robot_y = data_scenario["target_robot_y"]

        if not self.training:
            print(f"Scenario {random_scenario} loaded: target_robot_x={target_robot_x}, target_robot_y={target_robot_y}")
        

        self.humans_goals = jnp.array([[ [data_scenario["human1x"], data_scenario["human1y"]], [data_scenario["targethuman1x"], data_scenario["targethuman1y"]]],
                                    [ [data_scenario["human2x"], data_scenario["human2y"]], [data_scenario["targethuman2x"], data_scenario["targethuman2y"]]],
                                    [ [data_scenario["human3x"], data_scenario["human3y"]], [data_scenario["targethuman3x"], data_scenario["targethuman3y"]]],
                                    [ [data_scenario["human4x"], data_scenario["human4y"]], [data_scenario["targethuman4x"], data_scenario["targethuman4y"]]],
                                    [ [data_scenario["human5x"], data_scenario["human5y"]], [data_scenario["targethuman5x"], data_scenario["targethuman5y"]]]], dtype=jnp.float32)
        
        self.humans_current_goals = jnp.zeros((self.n_humans, 2), dtype=float)

        # Set human positions to their episode-specific starting positions
        humans_id = np.zeros(self.n_humans, dtype=int)
        for i in range(self.n_humans):
            humans_id[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            self.model.body_pos[humans_id[i], :2] = [data_scenario[f"human{i+1}x"], data_scenario[f"human{i+1}y"]]
            # Set human orientations
            self.model.body_quat[humans_id[i]] = [np.cos(data_scenario[f"start_orientation_human{i+1}"]/2), 0., 0., np.sin(data_scenario[f"start_orientation_human{i+1}"]/2)]




        # Set mobile robot and sphere positions
        self.data.qpos[:3] = [mob_robot_startposx, mob_robot_startposy, mob_robot_start_orientation]
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            self.model.geom_pos[sphere_geom_id, :] = [target_robot_x, target_robot_y, 2.0]

        # Reset velocities
        self.data.qvel[:] = 0

        # Update simulation
        mujoco.mj_forward(self.model, self.data)

        self.humans_state_numpy = np.zeros((self.n_humans, 6), dtype=float)

        # Set humans goal
        self.humans_current_goals = jnp.array(self.humans_goals[:, 0], dtype=jnp.float32)  # Start with the first goal for each human

        for i in range(self.n_humans):
            # Position
            self.humans_state_numpy[i, 0] = self.model.body_pos[humans_id[i], 0]  # x
            self.humans_state_numpy[i, 1] = self.model.body_pos[humans_id[i], 1]  # y
            
            # Velocity (initially zero)
            self.humans_state_numpy[i, 2] = 0.0  # vx
            self.humans_state_numpy[i, 3] = 0.0  # vy
            
            # Orientation from quaternion
            w, x, y, z = self.model.body_quat[humans_id[i]]
            theta_rad = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))    
            self.humans_state_numpy[i, 4] = theta_rad  # orientation
            
            # Angular velocity (initially zero)
            self.humans_state_numpy[i, 5] = 0.0  # angular velocity
            
            # Get observation and info
            observation = self._get_obs()
            info = self._get_info()
        
        return observation, info
    



    def step(self, action):
        self.time_of_the_episode = time.time() - self.episode_time_begin

        max_lin_vel = 1.0
        max_ang_vel = 1.0

        lin_vel = action[0] * max_lin_vel
        ang_vel = action[1] * max_ang_vel  
        #print(f"Action: lin_vel={lin_vel}, ang_vel={ang_vel}") 

        x, y, theta = self.data.qpos[:3]

        dt = self.model.opt.timestep # 0.1 [s/step]
        if (np.abs(ang_vel) > 1e-3):
            deltax = (lin_vel/ang_vel)*(np.sin(theta+ang_vel*dt) - np.sin(theta))
            deltay = (lin_vel/ang_vel)*(-np.cos(theta+ang_vel*dt) + np.cos(theta))
            x += deltax
            y += deltay
            theta += ang_vel*dt
        else:
            # Updating position
            x += lin_vel * np.cos(theta)*dt
            y += lin_vel * np.sin(theta)*dt
            # Updating orientation is not necessary, since we are not rotating
            

        # Set position and orientation
        self.data.qpos[:3] = [x, y, theta]
        
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        if self.all_obstacles:
            # Get all obstacles in the scene
            pass
        else:
            # recover grid position from humans positions
            found_obstacles = self.get_obstacles_from_human_positions(self.humans_state_numpy) # get obstacles from grid cell
            self.obstacles = self.get_static_obstacles_formatted(found_obstacles) # get corners from obstacles names

        # switch human target position to starting position if the first target is reached
        for i in range(self.n_humans):
            if np.linalg.norm(self.humans_state_numpy[i, :2] - self.humans_current_goals[i]) < 0.1 and self.humans_current_goals[i, 0] == self.humans_goals[i, 0, 0] and self.humans_current_goals[i, 1] == self.humans_goals[i, 0, 1]:
                #print(f"Human {i+1} reached goal, switching to starting position.")
                self.humans_current_goals = self.humans_current_goals.at[i].set(self.humans_goals[i, 1])

            elif np.linalg.norm(self.humans_state_numpy[i, :2] - self.humans_current_goals[i]) < 0.1 and self.humans_current_goals[i, 0] == self.humans_goals[i, 1, 0] and self.humans_current_goals[i, 1] == self.humans_goals[i, 1, 1]:
                self.humans_current_goals = self.humans_current_goals.at[i].set(self.humans_goals[i, 0]) # Switch to the first goal

        humans_state_jax = jnp.array(self.humans_state_numpy, dtype=jnp.float32)
        #Update humans
        for i in range(int(self.robot_dt / self.humans_dt)):
            next_humans_state = step(
                humans_state_jax,
                self.humans_current_goals,
                self.human_parameters,
                self.obstacles,
                self.humans_dt,
            )
            humans_state_jax = next_humans_state    

        #next_humans_state = humans_state_jax
        if np.isnan(next_humans_state).any():
            print("There are NaN values in the new state.")

        self.humans_state_numpy = np.array(next_humans_state)

        humans_id = np.zeros(self.n_humans, dtype=int)
        for i in range(self.n_humans):
            # Update human positions
            humans_id[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
            self.model.body_pos[humans_id[i], :] = [self.humans_state_numpy[i, 0], self.humans_state_numpy[i, 1], 0.0]
            # Update human orientations
            self.model.body_quat[humans_id[i]] = [np.cos(self.humans_state_numpy[i, 4]/2), 0., 0., np.sin(self.humans_state_numpy[i, 4]/2)]
        
        # Print target pos
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")

        mujoco.mj_forward(self.model, self.data)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        #print("info:", info)
        reward = 0.0

        reward += (self.previous_distance - np.linalg.norm(self.data.qpos[:2] - self.model.geom_pos[0, :2]))   # Reward for moving closer to the target
        self.previous_distance = np.linalg.norm(self.data.qpos[:2] - self.model.geom_pos[0, :2])  # Update previous distance
        self.episode_return += reward

        # Check for collisions
        too_close_to_obstacles = False
        for i in range(1, len(self.lidar_readings)):
            if self.lidar_readings[i] < (0.2+self.robot_radius) and self.lidar_readings[i] >= self.robot_radius:
                reward += -0.2/self.lidar_readings[i]
                #self.lidar_return += -0.2/self.lidar_readings[i]
                self.episode_return += -0.2/self.lidar_readings[i]
            if self.lidar_readings[i] < self.robot_radius:
                too_close_to_obstacles = True
                break  


        # Reward shaping part
        terminated = False
        truncated = False

        if too_close_to_obstacles:
            self.last_episode_result = "collision"
            terminated = True
            self.collision_counter += 1
            reward += -20.0
            self.episode_return += reward

        if self.time_of_the_episode > self.max_episode_time:
            self.last_episode_result = "timeout"
            truncated = True
            self.timeout_counter += 1
            reward = -10.0  # Negative reward for timeout
            self.episode_return += reward
            if not self.training:
                print(f"Episode timeout after {self.time_of_the_episode:.2f} seconds.")

        # If the target is reached terminate the episode
        if np.linalg.norm(self.data.qpos[:2] - self.model.geom_pos[0, :2]) < 0.1:
            self.last_episode_result = "success"
            terminated = True
            self.success_counter += 1
            reward += 200.0  # Positive reward for reaching the target
            self.episode_return += reward
            if not self.training:
                print(f"Target reached in {self.time_of_the_episode:.2f} seconds.")


        if self.render_mode == "human" and not self.training:
            self.render()

        info["episode_result"] = self.last_episode_result

        return observation, reward, terminated, truncated, info
    
    # def get_humans_state(self):
    #     humans_id = np.zeros(self.n_humans, dtype=int)
    #     for i in range(self.n_humans):
    #         humans_id[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"human{i+1}")
    #         self.humans_state[i, 0] = self.model.body_pos[humans_id[i], 0]
    #         self.humans_state[i, 1] = self.model.body_pos[humans_id[i], 1]

    #         angvel, linvel = self.data.cvel[humans_id[i], :3], self.data.cvel[humans_id[i], 3:6]
    #         self.humans_state[i, 2] = linvel[1]
    #         self.humans_state[i, 3] = linvel[0]

    #         w, x, y, z = self.model.body_quat[humans_id[i]]
    #         theta_rad = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))    
    #         theta_wrapped = (theta_rad + np.pi) % (2 * np.pi) - np.pi
    #         self.humans_state[i, 4] = theta_wrapped
    #         self.humans_state[i, 5] = angvel[2]
    #     return self.humans_state








    def get_obstacles_from_human_positions(self, humans_state):
        obstacles_per_human = []

        for i in range(self.n_humans):
            hx = humans_state[i, 0]
            hy = humans_state[i, 1]
            cell_x, cell_y = self.grid_cell_op.world_to_grid(hx, hy)

            try:
                with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid_cleaned.txt', 'r') as f:
                    lines = f.readlines()

                found_obstacles = []
                for line in lines:
                    # Tolleriamo sia "Cell x,y" che "Cell x, y" (con spazio)
                    if line.startswith(f"Cell {cell_x},{cell_y}") or line.startswith(f"Cell {cell_x}, {cell_y}"):
                        parts = line.strip().split(":", 1)
                        if len(parts) == 2:
                            obstacle_str = parts[1].strip()
                            if obstacle_str:
                                found_obstacles = obstacle_str.split("|")
                        break  # Trovata la riga, non serve continuare

                obstacles_per_human.append(found_obstacles)
                    
            except FileNotFoundError:
                print("File not found")
                return None

        return obstacles_per_human
    


    def get_static_obstacles_formatted(obstacle_names):
        static_obstacles = []

        try:
            with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/boxes_2d_corners.txt', 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.endswith(":"):
                    obstacle_name = line[:-1]
                    if obstacle_name in obstacle_names:
                        # Parse 4 vertices
                        vertices = []
                        for j in range(1, 5):
                            coord_line = lines[i + j].strip()
                            x_str, y_str = coord_line.split(',')
                            vertices.append([float(x_str.strip()), float(y_str.strip())])

                        # Create 4 edges: [v0,v1], [v1,v2], [v2,v3], [v3,v0] forn1 obstacles
                        edges = [
                            [vertices[0], vertices[1]],
                            [vertices[1], vertices[2]],
                            [vertices[2], vertices[3]],
                            [vertices[3], vertices[0]],
                        ]

                        # Pad with a dummy edge to ensure consistent shape
                        # nan_edge = [[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]
                        # edges.append(nan_edge)

                        static_obstacles.append(edges)
                        i += 4  # Skip the lines we've read

                i += 1

        except FileNotFoundError:
            print("boxes_2d_corners.txt not found")
            return None

        return jnp.array(static_obstacles)



    def get_all_obstacles(self):
        static_obstacles = []

        try:
            with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/boxes_2d_corners.txt', 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.endswith(":"):
                    # Parse 4 vertices
                    vertices = []
                    for j in range(1, 5):
                        coord_line = lines[i + j].strip()
                        x_str, y_str = coord_line.split(',')
                        vertices.append([float(x_str.strip()), float(y_str.strip())])

                    # Create 4 edges: [v0,v1], [v1,v2], [v2,v3], [v3,v0] forn1 obstacles
                    edges = [
                        [vertices[0], vertices[1]],
                        [vertices[1], vertices[2]],
                        [vertices[2], vertices[3]],
                        [vertices[3], vertices[0]],
                    ]

                    # Pad with a dummy edge to ensure consistent shape
                    # nan_edge = [[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]
                    # edges.append(nan_edge)

                    static_obstacles.append(edges)
                    i += 4  # Skip the lines we've read

                i += 1

        except FileNotFoundError:
            print("boxes_2d_corners.txt not found")
            return None

        return jnp.array(static_obstacles)
    
    def render(self, mode='human'):
        if self.training:
            return False

        if self.viewer is None:
            self._setup_viewer()
        
        if mode == 'human':
            self.viewer.sync()
            time.sleep(0.01)  # Small delay to control rendering speed
            return True
        return False
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    