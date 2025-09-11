import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer
# Scenario 6: DUE PORTE LATERALI con un umano

def scenario6_nh():
    rand_dim = 2.0
    max_x = 0.7
    random_x = np.random.uniform(-rand_dim, rand_dim)
    random_y = np.random.uniform(-rand_dim, rand_dim)
    rand_targ_x = np.random.uniform(-max_x, max_x)
    random_angle = np.random.uniform(-60, 60)
    rad_angle = np.deg2rad(random_angle)
    mob_robot_startposx = 55.16 + rand_targ_x
    mob_robot_startposy = 16.67

    in_rad_90 = np.deg2rad(-90)
    mob_robot_start_orientation = in_rad_90 + rad_angle #-90
    target_robot_x = 55.1 + rand_targ_x
    target_robot_y = 5.64

    human1x = -48.57 + random_x
    human1y = -9.2 + random_y
    start_orientation_human1 = 0
    targethuman1x = -60.55 + random_x
    targethuman1y = -10.2 + random_y

    human2x = -30
    human2y = -30
    start_orientation_human2 = 180
    targethuman2x = -0
    targethuman2y = -0

    human3x = -20
    human3y =  -20
    start_orientation_human3 = 90
    targethuman3x = 0
    targethuman3y = 0

    human4x = -10
    human4y = -10
    start_orientation_human4 = -90
    targethuman4x = -47.3
    targethuman4y = -4.08

    human5x = -40
    human5y = -40
    start_orientation_human5 = 180.0
    targethuman5x = -50.0
    targethuman5y = -50.0

    

    human6x = -30.38 + random_x
    human6y = -20.0 
    start_orientation_human6 = 0.0
    targethuman6x = -0.0 + random_x
    targethuman6y = -20.0 

    human7x = -29.26 + random_x
    human7y = -21.0 
    start_orientation_human7 = 0.0
    targethuman7x = -70.0 + random_x
    targethuman7y = -21.0

    human8x = -28.56 + random_x
    human8y = -19.0 
    start_orientation_human8 = 0.0
    targethuman8x = -70.0 + random_x
    targethuman8y = -22.0 

    human9x = -29.0 + random_x
    human9y = -19.0 
    start_orientation_human9 = 0.0
    targethuman9x = -70.0 + random_x
    targethuman9y = -23.0 

    human10x = -24.6 + random_x
    human10y = -19.0 
    start_orientation_human10 = 0.0
    targethuman10x = -70.0 + random_x
    targethuman10y = -24.0


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
        "human6x": human6x,
        "human6y": human6y,
        "start_orientation_human6": start_orientation_human6,
        "targethuman6x": targethuman6x,
        "targethuman6y": targethuman6y,
        "human7x": human7x,
        "human7y": human7y,
        "start_orientation_human7": start_orientation_human7,
        "targethuman7x": targethuman7x,
        "targethuman7y": targethuman7y,
        "human8x": human8x,
        "human8y": human8y,
        "start_orientation_human8": start_orientation_human8,
        "targethuman8x": targethuman8x,
        "targethuman8y": targethuman8y,
        "human9x": human9x,
        "human9y": human9y,
        "start_orientation_human9": start_orientation_human9,
        "targethuman9x": targethuman9x,
        "targethuman9y": targethuman9y,
        "human10x": human10x,
        "human10y": human10y,
        "start_orientation_human10": start_orientation_human10,
        "targethuman10x": targethuman10x,
        "targethuman10y": targethuman10y
    }






