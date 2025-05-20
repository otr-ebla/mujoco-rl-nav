import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

def scenario3():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    mob_robot_startposx = 30.77 + random_x
    mob_robot_startposy = -39 + random_y
    mob_robot_start_orientation = 90.0

    human1x = 36.39
    human1y = -27.77
    start_orientation_human1 = 90
    targethuman1x = 36.39
    targethuman1y = -16.214

    human2x = 22.38
    human2y = -27
    start_orientation_human2 = 0.0
    targethuman2x = 35.27
    targethuman2y = -26.93

    human3x = 34.35
    human3y = -23.4
    start_orientation_human3 = -90.0
    targethuman3x = 22.76
    targethuman3y = -23.4

    human4x = 0.0
    human4y = 0.0
    start_orientation_human4 = 90.0
    targethuman4x = 0.0
    targethuman4y = 0.0  

    human5x = 2.0
    human5y = 2.0
    start_orientation_human5 = -90.0
    targethuman5x = 2.0
    targethuman5y = 2.0

    target_robot_x = 49.57 + random_x
    target_robot_y = -22 + random_y

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






