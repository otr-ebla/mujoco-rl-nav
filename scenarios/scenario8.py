import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer
# Scenario 8: Infondo all'ambiente attraversa la porta, 3 umani nel mezzo



# NNNNNOOOOOOO

def scenario8():

    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-4.5, 4.5)
    random_angle = np.random.uniform(-40, 40)
    rad_angle = np.deg2rad(random_angle)
    mob_robot_startposx = 98.06
    mob_robot_startposy = 12.9

    in_rad_180 = np.deg2rad(180)
    mob_robot_start_orientation = in_rad_180 + rad_angle
    
    target_robot_x = 94.68
    target_robot_y = -8.4+random_y

    human1x = 95.35
    human1y = 0
    start_orientation_human1 = 90
    targethuman1x = 95.0
    targethuman1y = 19.0

    human2x = 100.0
    human2y = 2.0
    start_orientation_human2 = 180
    targethuman2x = 92.0
    targethuman2y = 2.0

    human3x = 92.12
    human3y = 11
    start_orientation_human3 = 0.0
    targethuman3x = 100.0
    targethuman3y = 11

    human4x = 2.0
    human4y = 2.0
    start_orientation_human4 = 90.0
    targethuman4x = 2.0
    targethuman4y = 2.0 

    human5x = 0.0
    human5y = 0.0
    start_orientation_human5 = -90.0
    targethuman5x = 0.0
    targethuman5y = 0.0

    

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






