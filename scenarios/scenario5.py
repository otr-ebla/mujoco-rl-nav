import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

# Scenario5: il robot entra nella prima porta a destra centrale e deve arrivare a attraversarne un'altra con dentro il target, 3 umani sono di mezzo

def scenario5():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    random_angle = np.random.uniform(-90, 90)
    mob_robot_startposx = 40.4
    mob_robot_startposy = -19 
    mob_robot_start_orientation = 90.0 + random_angle
    target_robot_x = 47.0 
    target_robot_y = 0 

    human1x = 44.5
    human1y = -12.7
    start_orientation_human1 = 180
    targethuman1x = 36.6
    targethuman1y = -12.7

    human2x = 43.86
    human2y = 7.7 + random_y
    start_orientation_human2 = -90.0
    targethuman2x = 42.86
    targethuman2y = -8 + random_y

    human3x = 37.52
    human3y = -9.6 + random_y
    start_orientation_human3 = 90.0
    targethuman3x = 37.52
    targethuman3y = 3.1 + random_y

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






