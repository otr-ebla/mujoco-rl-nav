import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer
# Scenario 4: Corridoio lungo

def scenarioTEST1_nh():
    random_x = np.random.uniform(-8, 8)
    random_xr = np.random.uniform(-4.5, 4.5)
    random_yr = np.random.uniform(-3.0, 3.0)
    random_y = np.random.uniform(-3.0, 3.0)

    randomy1 = np.random.uniform(-3.0, 3.0)
    randomy2 = np.random.uniform(-3.0, 3.0)
    randomy3 = np.random.uniform(-3.0, 3.0)
    randomy4 = np.random.uniform(-3.0, 3.0)
    randomy5 = np.random.uniform(-3.0, 3.0)

    randomy6 = np.random.uniform(-3.0, 3.0)
    randomy7 = np.random.uniform(-3.0, 3.0)
    randomy8 = np.random.uniform(-3.0, 3.0)
    randomy9 = np.random.uniform(-3.0, 3.0)
    randomy10 = np.random.uniform(-3.0, 3.0)

    random_angle = np.random.uniform(-45, 45)
    rad_angle = np.deg2rad(random_angle)
    
    mob_robot_startposx = 60.1 + random_xr
    mob_robot_startposy = 19 + random_yr

    in_rad_180 = np.deg2rad(180)
    mob_robot_start_orientation = in_rad_180 + rad_angle
    
    target_robot_x = 36.9 + random_xr
    target_robot_y = 20 + random_yr



    # humans
    # red
    human1x = -44.56 + random_x
    human1y = -19.43 + randomy1
    start_orientation_human1 = 0
    targethuman1x = -71.1+ random_x
    targethuman1y = -16.21 + randomy1

    human2x = -41.0 + random_x
    human2y = -20.5 + randomy2
    start_orientation_human2 = 0
    targethuman2x = -70.4+random_x
    targethuman2y = 20.6 + randomy2

    human3x = -40.61 + random_x
    human3y = -18.8 + randomy3
    start_orientation_human3 = 0.0
    targethuman3x = -70.08 + random_x
    targethuman3y = 18.8 + randomy3

    # pink
    human4x = -36.6 + random_x
    human4y = -19.0 + randomy4
    start_orientation_human4 = 0.0
    targethuman4x = -70.1+random_x
    targethuman4y = -16.0 + randomy4

    human5x = -38.1 + random_x
    human5y = -19.8 + randomy5
    start_orientation_human5 = 0.0
    targethuman5x = -59.0 + random_x
    targethuman5y = -19.8 + randomy5

    human6x = -30.38 + random_x
    human6y = -20.0 + randomy6
    start_orientation_human6 = 0.0
    targethuman6x = 70.0 + random_x
    targethuman6y = -20.0 + randomy6

    human7x = -29.26 + random_x
    human7y = -21.0 + randomy7
    start_orientation_human7 = 0.0
    targethuman7x = -70.0 + random_x
    targethuman7y = -21.0 + randomy7

    human8x = -28.56 + random_x
    human8y = -19.0 + randomy8
    start_orientation_human8 = 0.0
    targethuman8x = -70.0 + random_x
    targethuman8y = -22.0 + randomy8

    human9x = -29.0 + random_x
    human9y = -19.0 + randomy9
    start_orientation_human9 = 0.0
    targethuman9x = -70.0 + random_x
    targethuman9y = -23.0 + randomy9

    human10x = -24.6 + random_x
    human10y = -19.0 + randomy10
    start_orientation_human10 = 0.0
    targethuman10x = -70.0 + random_x
    targethuman10y = 24.0 + randomy10


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
        "targethuman10y": targethuman10y,
    }






