import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer
# Scenario 6: il robot deve girare attorno al tavolo centrale

# IMPOSSIBILE

def scenario6():
    rand_dim = 2.0
    random_x = np.random.uniform(-rand_dim, rand_dim)
    random_y = np.random.uniform(-rand_dim, rand_dim)
    random_angle = np.random.uniform(-60, 60)
    rad_angle = np.deg2rad(random_angle)
    mob_robot_startposx = 55.16 + random_x
    mob_robot_startposy = 16.67

    in_rad_90 = np.deg2rad(-90)
    mob_robot_start_orientation = in_rad_90 + rad_angle #-90
    target_robot_x = 55.33 + random_x
    target_robot_y = 6.64

    human1x = 48.57 + random_x
    human1y = 9.2 + random_y
    start_orientation_human1 = 0
    targethuman1x = 60.55 + random_x
    targethuman1y = 10.2 + random_y

    human2x = 3
    human2y = 3
    start_orientation_human2 = 180
    targethuman2x = 0
    targethuman2y = 0

    human3x = 2
    human3y =  2
    start_orientation_human3 = 90
    targethuman3x = 0
    targethuman3y = 0

    human4x = 1
    human4y = 1
    start_orientation_human4 = -90
    targethuman4x = 47.3
    targethuman4y = -4.08

    human5x = 4
    human5y = 4
    start_orientation_human5 = 180.0
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






