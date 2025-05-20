import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

def scenario6():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    mob_robot_startposx = 96.1 + random_x
    mob_robot_startposy = 40.86 + random_y
    mob_robot_start_orientation = 180
    target_robot_x = 29.21 + random_x
    target_robot_y = 40.86 + random_y

    human1x = 29.21
    human1y = 43.46
    start_orientation_human1 = 0
    targethuman1x = 88.66
    targethuman1y = 43.46

    human2x = 23.03
    human2y = 37.1
    start_orientation_human2 = 0
    targethuman2x = 91.9
    targethuman2y = 37.1

    human3x = 28.75
    human3y = 33.55
    start_orientation_human3 = 0.0
    targethuman3x = 88.01
    targethuman3y = 33.55

    human4x = 70
    human4y = -9.0
    start_orientation_human4 = 0.0
    targethuman4x = 48.0
    targethuman4y = -9.0 

    human5x = 86.96
    human5y = 45.63
    start_orientation_human5 = 180.0
    targethuman5x = 33.49
    targethuman5y = 45.63

    

    # return data
    return {
        "mob_robot_startposx": mob_robot_startposx,
        "mob_robot_startposy": mob_robot_startposy,
        "mob_robot_start_orientation": mob_robot_start_orientation,
        "target_robot_x": target_robot_x,
        "target_robot_y": target_robot_y,
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






