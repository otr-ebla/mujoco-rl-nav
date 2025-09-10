import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym 
import mujoco
import mujoco.viewer

# Scenario 1: Il robot attraversa il corridoio centrale iniziale e raggiunge il target al fondo davanti alla parete
#             Gli umani si incrociano lì davanti

def scenarioTEST3():
    random_x = np.random.uniform(0, 6)
    random_y = np.random.uniform(-4, 4)

    random_angle = np.random.uniform(-45, 45)
    rad_angle = np.deg2rad(random_angle)

    mob_robot_startposx = 17.26 + random_x
    mob_robot_startposy = 0
    init_angle = 180
    init_rad = np.deg2rad(init_angle)
    mob_robot_start_orientation = init_rad + rad_angle
    target_robot_x = 2.26
    target_robot_y = 0 + random_y

    human1x = 14.36
    human1y = -3.36
    start_orientation_human1 = 90
    targethuman1x = 14.36
    targethuman1y = 2.73

    human2x = 12.5
    human2y = 2.73
    start_orientation_human2 = -90
    targethuman2x = 12.5
    targethuman2y = -3.36

    human3x = 9
    human3y = -3.36
    start_orientation_human3 = 90.0
    targethuman3x = 9
    targethuman3y = 2.73

    human4x = 6.5
    human4y = 2.73
    start_orientation_human4 = -90.0
    targethuman4x = 6.5
    targethuman4y = -3.36

    human5x = 3.85
    human5y = -3.36
    start_orientation_human5 = 90.0
    targethuman5x = 3.85
    targethuman5y = 2.73



    human6x = 10.5
    human6y = 2.73
    start_orientation_human6 = 90.0
    targethuman6x = 10.5
    targethuman6y = -3.36

    human7x = 5
    human7y = 2.73
    start_orientation_human7 = -90.0
    targethuman7x = 5
    targethuman7y = -3.36

    human8x = 28.56 + random_x
    human8y = 9.29 + random_y 
    start_orientation_human8 = -90
    targethuman8x = 28.56 + random_x
    targethuman8y = -4.9 + random_y

    human9x = -29.0 + random_x
    human9y = 19.0 
    start_orientation_human9 = 0.0
    targethuman9x = -70.0 + random_x
    targethuman9y = 23.0 

    human10x = -24.6 + random_x
    human10y = 19.0 
    start_orientation_human10 = 0.0
    targethuman10x = -70.0 + random_x
    targethuman10y = 24.0


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






