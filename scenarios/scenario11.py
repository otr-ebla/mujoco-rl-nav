import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer
# Scenario11: simile a scenario5 il robot deve entrare dentro una porta senza umani nelle vicinanze

def scenario11():
    random_x = np.random.uniform(-3.5, 3.5)
    random_y = np.random.uniform(-2.5, 2.5)
    random_angle = np.random.uniform(-35, 35)
    rad_angle = np.deg2rad(random_angle)    
    mob_robot_startposx = 40.4 + random_x
    mob_robot_startposy = -19

    in_rad_90 = np.deg2rad(90.0)
    mob_robot_start_orientation = in_rad_90 + rad_angle
    
    target_robot_x = 39.6 + random_x # randomy per renderlo diverso da randomx
    target_robot_y = -11.78

    human1x = 4
    human1y = -4
    start_orientation_human1 = 180
    targethuman1x = 5
    targethuman1y = -5

    human2x = 5
    human2y = 5
    start_orientation_human2 = -90.0
    targethuman2x = 5
    targethuman2y = 5

    human3x = 6
    human3y = 6
    start_orientation_human3 = 90.0
    targethuman3x = 6
    targethuman3y = 6

    human4x = -2.0
    human4y = -2.0
    start_orientation_human4 = 180
    targethuman4x = -2.
    targethuman4y = -2. 

    human5x = 2.0
    human5y = 2.0
    start_orientation_human5 = -90.0
    targethuman5x = 2.0
    targethuman5y = 2.0

    

    # return data
    return {
        "mob_robot_startposx": mob_robot_startposx,
        "mob_robot_startposy": mob_robot_startposy,
        "mob_robot_start_orientation": mob_robot_start_orientation,
        "target_robot_x": target_robot_x,
        "target_robot_y": target_robot_y,
        "rad_angle": rad_angle,
        "human1x": human1x,
        "human1y": human1y,
        "start_orientation_human1": start_orientation_human1,
        "targethuman1x": targethuman1x,
        "targethuman1y": targethuman1y,
        "human2x": human2x,
        "human2y": human2y,
        "start_orientation_human2": start_orientation_human2,
        "targethuman2x": targethuman2x,
        "targethuman2y": targethuman2y,
        "human3x": human3x,
        "human3y": human3y,
        "start_orientation_human3": start_orientation_human3,
        "targethuman3x": targethuman3x,
        "targethuman3y": targethuman3y,
        "human4x": human4x,
        "human4y": human4y,
        "start_orientation_human4": start_orientation_human4,
        "targethuman4x": targethuman4x,
        "targethuman4y": targethuman4y,
        "human5x": human5x,
        "human5y": human5y,
        "start_orientation_human5": start_orientation_human5,
        "targethuman5x": targethuman5x,
        "targethuman5y": targethuman5y,
    }






