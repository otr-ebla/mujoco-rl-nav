import numpy as np
import gymnasium as gym 
import mujoco
import mujoco_viewer
from mobilerobotRL import mobilerobotRL
import os
import xml.etree.ElementTree as ET
import time
import random
from scenarios.scenario1 import scenario1, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8 

class hamrrln(mobilerobotRL):
    def __init__(self, num_rays = 108, model_path = "assets/world.xml") -> None:
        super().__init__()

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


        # Reset the environment
        # Choose a random scenario
        # Generate a random integer between 1 and 8
        scenario_switch = {
            1: scenario1,
            2: scenario2,
            3: scenario3,
            4: scenario4,
            5: scenario5,
            6: scenario6,
            7: scenario7,
            8: scenario8
        }

        random_scenario = random.randint(1, 8)
        data_scenario = scenario_switch.get(random_scenario, lambda: None())()




        # Set mobile robot and sphere positions
        self.data.qpos[:3] = [cube_x, cube_y, cube_yaw]
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            self.model.geom_pos[sphere_geom_id, :] = [sphere_x, sphere_y, 2.0]

        # Reset velocities
        self.data.qvel[:] = 0
        
        # Update simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self):
        pass

    

    