import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

def scenario7():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    mob_robot_startposx = 4.31 + random_x
    mob_robot_startposy = -16.08 + random_y
    mob_robot_start_orientation = 90
    target_robot_x = 4.31 + random_x
    target_robot_y = 9.7 + random_y

    human1x = 11.23
    human1y = -11.23
    start_orientation_human1 = 133
    targethuman1x = -4.22
    targethuman1y = 5.14

    human2y = 10.8
    human2x = 8.57
    start_orientation_human2 = -120
    targethuman2x = -6.15
    targethuman2y = -10.19

    human3x = -12.6
    human3y = 0.0
    start_orientation_human3 = 0.0
    targethuman3x = 9.45
    targethuman3y = 0.0

    human4x = -2.32
    human4y = -12.43
    start_orientation_human4 = 90.0
    targethuman4x = -2.32
    targethuman4y = 7.7  

    human5x = 3.0
    human5y = 9.44
    start_orientation_human5 = -90.0
    targethuman5x = 8.32
    targethuman5y = -13.33

    

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






