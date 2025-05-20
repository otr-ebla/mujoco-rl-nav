import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

def scenario5():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    mob_robot_startposx = 60. + random_x
    mob_robot_startposy = -39 + random_y
    mob_robot_start_orientation = -90.0
    target_robot_x = 60.0 + random_x
    target_robot_y = -19.27 + random_y

    human1x = 44.25
    human1y = 10.18
    start_orientation_human1 = 0
    targethuman1x = 74.7
    targethuman1y = 10.18

    human2x = 74.7
    human2y = 8.1
    start_orientation_human2 = -90.0
    targethuman2x = 74.7
    targethuman2y = -9.0

    human3x = 45.78
    human3y = -9.01
    start_orientation_human3 = 90.0
    targethuman3x = 45.78
    targethuman3y = 9.43

    human4x = 70
    human4y = -9.0
    start_orientation_human4 = 180
    targethuman4x = 48.0
    targethuman4y = -9.0 

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






