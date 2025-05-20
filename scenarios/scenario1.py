import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

def scenario1():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    mob_robot_startposx = -20 + random_x
    mob_robot_startposy = 0 + random_y
    
    human1x = -32.0
    human1y = 6.4
    start_orientation_human1 = 0.0
    targethuman1x = 9.35
    targethuman1y = 6.4

    human2x = 9.35
    human2y = -6.4
    start_orientation_human2 = 0.0
    targethuman2x = -32.
    targethuman2y = -6.4

    human3x = -5.7
    human3y = 17.1
    start_orientation_human3 = -90.0
    targethuman3x = -5.7
    targethuman3y = -17.1

    human4x = 0.0
    human4y = -17.1
    start_orientation_human4 = 90.0
    targethuman4x = 0
    targethuman4y = 17.1    

    human5x = 5.8
    human5y = 17.58
    start_orientation_human5 = -90.0
    targethuman5x = 5.8
    targethuman5y = -12.40

    target_robot_x = -5.7
    target_robot_y = 17.1 + random_y

    # return data
    return {
        "mob_robot_startposx": mob_robot_startposx,
        "mob_robot_startposy": mob_robot_startposy,
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
        "target_robot_x": target_robot_x,
        "target_robot_y": target_robot_y
    }






