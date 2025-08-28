import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer
# Scenario 4: Corridoio lungo

def scenario4():
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-3.0, 3.0)
    random_angle = np.random.uniform(-45, 45)
    rad_angle = np.deg2rad(random_angle)
    
    mob_robot_startposx = 60.1 + random_x
    mob_robot_startposy = 19 + random_y

    in_rad_90 = np.deg2rad(90)
    mob_robot_start_orientation = in_rad_90 + rad_angle
    
    target_robot_x = 41.9 + random_x
    target_robot_y = 20 + random_y

    human1x = 71.1+ random_x
    human1y = 17.43
    start_orientation_human1 = 180
    targethuman1x = 44.56+ random_x
    targethuman1y = 16.21

    human2x = 70.4+random_x
    human2y = 20.5
    start_orientation_human2 = 180
    targethuman2x = 41.0+ random_x
    targethuman2y = 20.6

    human3x = 40.61
    human3y = 18.8
    start_orientation_human3 = 0.0
    targethuman3x = 70.08
    targethuman3y = 18.8

    human4x = 36.6 + random_x
    human4y = 16.0
    start_orientation_human4 = 0.0
    targethuman4x = 70.1+random_x
    targethuman4y = 16.0  

    human5x = 30.1 
    human5y = 19.8 +random_y
    start_orientation_human5 = 0.0
    targethuman5x = 49.0 + random_x
    targethuman5y = 19.8 + random_y

    

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






