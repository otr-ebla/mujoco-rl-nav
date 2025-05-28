import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from torch.utils.tensorboard import SummaryWriter
import xml.etree.ElementTree as ET  

class mobilerobotRL(gym.Env):
    def __init__(self, num_rays = 108, training = True, log_dir="TENSORBOARD", model_path = "assets/world.xml") -> None:
        super().__init__()  
        self.training = training
        self.num_rays = num_rays
        self.max_episode_steps = 1000
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
        self.render_mode = None
        self.lidar_readings = None  
        self.success_rate = 0
        self.collision_rate = 0
        self.timeout_rate = 0
        self.robot_relative_azimuth = 0
        self.model_path = model_path
        self.training_mode = training

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
        if self.render_mode == "human":
            self._setup_viewer()

        self.reset()

    def load_and_modify_xml_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        
        tree = ET.parse(self.model_path)
        root = tree.getroot()

        mobile_robot_body = None
        for body in root.findall('.//body'):
            if body.get("name") == "agent_body":
                mobile_robot_body = body
                break
        
        if mobile_robot_body is None:
            raise ValueError("Mobile robot body not found in the XML model.")
        
        sensor = None
        for s in root.findall('.//sensor'):
            sensor = s
            break

        if sensor is None:
            raise ValueError("Sensor not found in the XML model.")
        
        # Add lidar rangefinder sensor to the mobile robot body
        for i in range(self.num_rays):
            angle = (i / self.num_rays) * 2 * np.pi

            #angle = (-np.pi / self.num_rays) * i + np.pi / 2  
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            site = ET.SubElement(mobile_robot_body, 'site')
            site.set('name', f"lidar_site_{i}")
            site.set('pos', f"{0.0} {-0.05} -0.3")
            site.set('size', "0.05")
            site.set('rgba', "1 0 0 1")
            site.set("zaxis", f"{cos_angle} {sin_angle} 0")

            rangefinder = ET.Element("rangefinder")
            rangefinder.set("name", f"lidar_{i}")
            rangefinder.set("site", f"lidar_site_{i}")
            sensor.append(rangefinder)

        return ET.tostring(root, encoding='unicode')
    
    def _setup_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 25.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.lookat[:] = [0, 0, 1]

    def _get_obs(self):
        
        self.lidar_readings = np.array([self.data.sensordata[lidar_id] for lidar_id in self.lidar_sensor_ids])
        self.lidar_readings = self.lidar_readings.flatten()

        agent_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        self.robot_pos = self.data.xpos[agent_body_id].copy()

        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        #print(f"TARGET POS: {self.data.geom_xpos[sphere_geom_id]}")
        self.target_pos = self.data.geom_xpos[sphere_geom_id].copy()
        self.robot_rot_matrix = self.data.xmat[agent_body_id].reshape(3, 3)

        robot_forward_vector = self.robot_rot_matrix[:, 0]
        robot_yaw_angle = np.arctan2(robot_forward_vector[1], robot_forward_vector[0])

        relative_position = self.target_pos - self.robot_pos
        distance_target_robot = np.linalg.norm(relative_position[:2])
        global_robot_azimuth = np.arctan2(relative_position[1], relative_position[0])

        self.robot_relative_azimuth = global_robot_azimuth - robot_yaw_angle
        self.robot_relative_azimuth = (self.robot_relative_azimuth+np.pi)%(2*np.pi)-np.pi

        obs = np.concatenate((self.lidar_readings, [distance_target_robot, self.robot_relative_azimuth]))
        return obs.astype(np.float32)
    
    def _get_info(self):
        agent_pos = np.zeros(3)
        sphere_pos = np.zeros(3)

        agent_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")  
        if agent_body_id >= 0:
            agent_pos = self.data.xpos[agent_body_id].copy()

        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()
        
        distance_to_sphere = np.linalg.norm(sphere_pos - agent_pos)
        #print(f"Distance to sphere: {distance_to_sphere:.2f}")
        return {
            "distance_to_sphere": distance_to_sphere,
            "cube_position": agent_pos,
            "sphere_position": sphere_pos
        }

    

    def _get_info(self):
        # Get positions using direct geom access instead of sensors
        cube_pos = np.zeros(3)
        sphere_pos = np.zeros(3)
        
        # Get cube position from its body
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        if cube_body_id >= 0:
            # Get position of the cube body in world coordinates
            cube_pos = self.data.xpos[cube_body_id].copy()
        
        # Get sphere position directly from its geom
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            # Get position of the sphere geom in world coordinates
            sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()
        
        # Calculate distance to sphere
        distance_to_sphere = np.linalg.norm(sphere_pos - cube_pos)

        
        
        return {
            "distance_to_sphere": distance_to_sphere,
            "robot_position": cube_pos,
            "target_position": sphere_pos
        }


    def reset(self, seed=None, options=None):
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

        if self.last_episode_result == "success" and self.training_mode == False:
            
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
        self.previous_distance = 30  # Reset distance tracking
        self.stuck_counter = 0  # Reset stuck counter










        # Reset velocities
        self.data.qvel[:] = 0
        
        # Update simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.episode_time_length = time.time() - self.episode_time_begin
        move_max_lin_speed = 0.25
        max_ang_speed = 1.0

        linear_velocity = action[0] * move_max_lin_speed
        angular_velocity = action[1] * max_ang_speed

        x, y, theta = self.data.qpos[:3]
        dt = self.model.opt.timestep # 0.1 [s/step]
        if (np.abs(angular_velocity) > 1e-3):
            deltax = (linear_velocity/angular_velocity)*(np.sin(theta+angular_velocity*dt) - np.sin(theta))
            deltay = (linear_velocity/angular_velocity)*(-np.cos(theta+angular_velocity*dt) + np.cos(theta))
            x += deltax
            y += deltay
            theta += angular_velocity*dt
        else:
            # Updating position
            x += linear_velocity * np.cos(theta)*dt
            y += linear_velocity * np.sin(theta)*dt
            # Updating orientation is not necessary, since we are not rotating
            

        # Set position and orientation
        self.data.qpos[:3] = [x, y, theta]
        
        # Step simulation
        while self.data.time < 100:
            mujoco.mj_step(self.model, self.data)  # Advances simulation by one step
        
        self.current_step += 1
        
        # Update simulation state
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        reward = 0.0
        # Check for collision with obstacles
        contact_with_obstacles = False
        contact_with_sphere = False

        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            if geom1_id == cube_geom_id or geom2_id == cube_geom_id:
                # Check if the contact is with the walls
                other_geom_id = geom2_id if geom1_id == cube_geom_id else geom1_id

                other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id)

                if (other_geom_name and "wall" in other_geom_name) or (other_geom_name and "obstacle" in other_geom_name) :  
                    contact_with_obstacles = True
                    break
                elif other_geom_name and "sphere" in other_geom_name:
                    contact_with_sphere = True
                    break
        
        # Calculate reward and check termination conditions
        distance_to_sphere = info["distance_to_sphere"]

    
        reward += 1.5*(self.previous_distance - distance_to_sphere)
        
        reward += -0.1*abs(self.relative_azimuth)

        
        # Terminal conditions
        terminated = False
        truncated = False


        too_close_to_obstacles = False
        for i in range(1, len(self.lidar_readings)):
            if self.lidar_readings[i] < (0.2+self.robot_radius) and self.lidar_readings[i] > (0.01+self.robot_radius):
                reward += -0.1/self.lidar_readings[i]
                
            elif self.lidar_readings[i] <= (0.01+self.robot_radius):
                too_close_to_obstacles = True
                break

        self.episode_return += reward           
        
        if contact_with_obstacles or too_close_to_obstacles:
            reward += -20.0
            self.episode_return += reward
            self.last_episode_result = "collision"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin
            terminated = True   

        elif distance_to_sphere < 2.0 or contact_with_sphere:  
            reward += 200
            self.episode_return += reward
            self.last_episode_result = "success"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin
            terminated = True

        elif self.current_step >= self.max_episode_steps:
            #reward = -50.0
            #self.episode_return += reward
            self.last_episode_result = "timeout"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin
            truncated = True

        
        # if np.abs(self.previous_distance - distance_to_sphere) < 0.01 and not self.training_mode:
        #     self.stuck_counter += 1
        #     if self.stuck_counter >= 100:
        #         reward = -10.0
        #         self.episode_return += reward
        #         self.last_episode_result = "timeout"
        #         info["steps_taken"] = self.current_step
        #         info["episode_time_length"] = time.time() - self.episode_time_begin
        #         terminated = True
        
        self.previous_distance = distance_to_sphere

        info["episode_result"] = self.last_episode_result

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
