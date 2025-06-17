import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco_viewer

# Scenario 1: Il robot attraversa il corridoio centrale iniziale e raggiunge il target al fondo davanti alla parete
#             Gli umani si incrociano lì davanti

def scenario1_nh():
    delta_random = 3
    random_x = np.random.uniform(-delta_random, delta_random)
    random_y = np.random.uniform(-delta_random, delta_random)
    random_angle = np.random.uniform(-90, 90)
    mob_robot_startposx = 16.85 + random_x
    mob_robot_startposy = 0 + random_y
    mob_robot_start_orientation = 0 + random_angle
    target_robot_x = 32.8
    target_robot_y = 0.0 + random_y
    
    human1x = -6
    human1y = 0
    start_orientation_human1 = 0.0
    targethuman1x = -6
    targethuman1y = 0

    human2x = -6
    human2y = 0
    start_orientation_human2 = -20
    targethuman2x = -6.2
    targethuman2y = 0

    human3x = -5.3
    human3y = 0
    start_orientation_human3 = -150
    targethuman3x = -5.3
    targethuman3y = 0

    human4x = -6.3
    human4y = 0
    start_orientation_human4 = 90.0
    targethuman4x = -6.3
    targethuman4y = 0    

    human5x = -6.4
    human5y = -1
    start_orientation_human5 = -90.0
    targethuman5x = -6.4
    targethuman5y = -1

    

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






