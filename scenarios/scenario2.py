import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer

# Scenario 2: Uguale a scenario1, ma il robot deve girare a destra e raggiungere il target

def scenario2():
    delta_random = 3
    random_x = np.random.uniform(-delta_random, delta_random)
    random_y = np.random.uniform(-delta_random, delta_random)
    random_angle = np.random.uniform(-40, 40)
    rad_angle = np.deg2rad(random_angle)
    mob_robot_startposx = 19.85 + random_x
    mob_robot_startposy = 0 + random_y
    
    mob_robot_start_orientation = 0 + rad_angle

    target_robot_x = 29.63
    target_robot_y = -9.5 + random_y
    
    human1x = 27.43 + random_x
    human1y = -8.7 + random_y
    start_orientation_human1 = 0.0
    targethuman1x = 27.43
    targethuman1y = 7.63

    human2x = 21.67
    human2y = -9
    start_orientation_human2 = -20
    targethuman2x = 32
    targethuman2y = 6.74

    human3x = 23
    human3y = 6.74
    start_orientation_human3 = -150
    targethuman3x = 32.4
    targethuman3y = -6.8

    human4x = 32.3
    human4y = -2.6
    start_orientation_human4 = 90.0
    targethuman4x = 16.26
    targethuman4y = -2.6    

    human5x = 16.26
    human5y = 2.77
    start_orientation_human5 = -90.0
    targethuman5x = 29.45 + random_x
    targethuman5y = 2.77 + random_y


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






