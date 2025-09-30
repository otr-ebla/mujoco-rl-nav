# import numpy as np

# # Scenario 9: Perpendicular traffic

# def scenario9():

#     random_x = np.random.uniform(0, 6)
#     random_y = np.random.uniform(-4, 4)

#     random_angle = np.random.uniform(-180, 180)
#     rad_angle = np.deg2rad(random_angle)

#     mob_robot_startposx = 17.26 + random_x
#     mob_robot_startposy = 0
#     init_angle = 180
#     init_rad = np.deg2rad(init_angle)
#     mob_robot_start_orientation = init_rad + rad_angle
#     target_robot_x = 2.26
#     target_robot_y = 0 + random_y

#     human1x = 14.36
#     human1y = -4.5
#     start_orientation_human1 = 90
#     targethuman1x = 14.36
#     targethuman1y = 4.0

#     human2x = 12.5
#     human2y = 4.0
#     start_orientation_human2 = -90
#     targethuman2x = 12.5
#     targethuman2y = -4.5

#     human3x = 9
#     human3y = -4.5
#     start_orientation_human3 = 90.0
#     targethuman3x = 9
#     targethuman3y = 4.0

#     human4x = 6.5
#     human4y = 4.0
#     start_orientation_human4 = -90.0
#     targethuman4x = 6.5
#     targethuman4y = -4.5

#     human5x = 3.85
#     human5y = -4.5
#     start_orientation_human5 = 90.0
#     targethuman5x = 3.85
#     targethuman5y = 4.0

    

#     human6x = -30.38 + random_x
#     human6y = 20.0 
#     start_orientation_human6 = 0.0
#     targethuman6x = -0.0 + random_x
#     targethuman6y = 20.0 

#     human7x = -29.26 + random_x
#     human7y = 21.0 
#     start_orientation_human7 = 0.0
#     targethuman7x = -70.0 + random_x
#     targethuman7y = 21.0 

#     human8x = -28.56 + random_x
#     human8y = 19.0 
#     start_orientation_human8 = 0.0
#     targethuman8x = -70.0 + random_x
#     targethuman8y = 22.0 

#     human9x = -29.0 + random_x
#     human9y = 19.0 
#     start_orientation_human9 = 0.0
#     targethuman9x = -70.0 + random_x
#     targethuman9y = 23.0 

#     human10x = -24.6 + random_x
#     human10y = 19.0 
#     start_orientation_human10 = 0.0
#     targethuman10x = -70.0 + random_x
#     targethuman10y = 24.0


#     # return data
#     return {
#         "mob_robot_startposx": mob_robot_startposx,
#         "mob_robot_startposy": mob_robot_startposy,
#         "mob_robot_start_orientation": mob_robot_start_orientation,
#         "target_robot_x": target_robot_x,
#         "target_robot_y": target_robot_y,
#         "rad_angle": rad_angle,
#         "human1x": human1x,
#         "human1y": human1y,
#         "start_orientation_human1": start_orientation_human1,
#         "targethuman1x": targethuman1x,
#         "targethuman1y": targethuman1y,
#         "human2x": human2x,
#         "human2y": human2y,
#         "start_orientation_human2": start_orientation_human2,
#         "targethuman2x": targethuman2x,
#         "targethuman2y": targethuman2y,
#         "human3x": human3x,
#         "human3y": human3y,
#         "start_orientation_human3": start_orientation_human3,
#         "targethuman3x": targethuman3x,
#         "targethuman3y": targethuman3y,
#         "human4x": human4x,
#         "human4y": human4y,
#         "start_orientation_human4": start_orientation_human4,
#         "targethuman4x": targethuman4x,
#         "targethuman4y": targethuman4y,
#         "human5x": human5x,
#         "human5y": human5y,
#         "start_orientation_human5": start_orientation_human5,
#         "targethuman5x": targethuman5x,
#         "targethuman5y": targethuman5y,
#         "human6x": human6x,
#         "human6y": human6y,
#         "start_orientation_human6": start_orientation_human6,
#         "targethuman6x": targethuman6x,
#         "targethuman6y": targethuman6y,
#         "human7x": human7x,
#         "human7y": human7y,
#         "start_orientation_human7": start_orientation_human7,
#         "targethuman7x": targethuman7x,
#         "targethuman7y": targethuman7y,
#         "human8x": human8x,
#         "human8y": human8y,
#         "start_orientation_human8": start_orientation_human8,
#         "targethuman8x": targethuman8x,
#         "targethuman8y": targethuman8y,
#         "human9x": human9x,
#         "human9y": human9y,
#         "start_orientation_human9": start_orientation_human9,
#         "targethuman9x": targethuman9x,
#         "targethuman9y": targethuman9y,
#         "human10x": human10x,
#         "human10y": human10y,
#         "start_orientation_human10": start_orientation_human10,
#         "targethuman10x": targethuman10x,
#         "targethuman10y": targethuman10y
#     }

import numpy as np

# Scenario 9: Perpendicular traffic (randomized, corridor-safe)
def scenario9(seed=None):
    if seed is not None:
        np.random.seed(seed)

    # --- Corridor geometry (adjust if your corridor differs) ---
    # Y bounds of the corridor where pedestrians cross (keep them inside)
    Y_MIN, Y_MAX = -4.5, 4.0
    Y_MARGIN = 0.2           # safety margin from walls
    y_low  = Y_MIN + Y_MARGIN
    y_high = Y_MAX - Y_MARGIN

    # Robot path is roughly along +x -> -x in your original file
    X_ENTRY  = 17.26         # where the robot starts (base)
    X_TARGET =  2.26         # where the robot must arrive

    # --- Robot start / target (with randomness similar to your original) ---
    random_x = np.random.uniform(0, 6.0)
    random_y = np.random.uniform(-4.0, 4.0)
    random_angle = np.random.uniform(-45.0, 45.0)
    rad_angle = np.deg2rad(random_angle)

    mob_robot_startposx = X_ENTRY + random_x
    mob_robot_startposy = 0.0
    init_angle = 180.0
    init_rad = np.deg2rad(init_angle)
    mob_robot_start_orientation = init_rad + rad_angle

    target_robot_x = X_TARGET
    # teniamo il target y variabile ma dentro corridoio con margine
    target_robot_y = float(np.clip(random_y, y_low, y_high))

    # --- Where to place the 5 crossing pedestrians (x positions) ---
    # Keep them between robot target and start, with spacing.
    n_crossers = 5
    x_min = target_robot_x + 1.0
    x_max = mob_robot_startposx - 2.0
    if x_max <= x_min + 1.0:
        # fallback if start is too close to target due to randomness
        x_max = x_min + 3.0

    def sample_sorted_positions(n, lo, hi, min_sep):
        xs = []
        tries = 0
        while len(xs) < n and tries < 2000:
            c = np.random.uniform(lo, hi)
            if all(abs(c - v) >= min_sep for v in xs):
                xs.append(c)
            tries += 1
        if len(xs) < n:
            xs = list(np.linspace(lo, hi, n))
        xs.sort()
        return xs

    # min separation proportional to available span
    span = max(hi := x_max, lo := x_min) - min(hi, lo)
    min_sep = max(0.8, 0.15 * max(x_max - x_min, 1.0))
    x_positions = sample_sorted_positions(n_crossers, x_min, x_max, min_sep)

    # --- Randomize start sides and slight jitter on endpoints ---
    def rand_edge_pair():
        # Randomly choose who starts at bottom or top; add small jitter so paths differ
        start_top = np.random.rand() < 0.5
        jitter = np.random.uniform(0.0, 0.35)
        if start_top:
            y_start = y_high - jitter
            y_goal  = y_low  + jitter
            start_theta = -90.0  # heading downwards
        else:
            y_start = y_low  + jitter
            y_goal  = y_high - jitter
            start_theta = 90.0   # heading upwards
        return y_start, y_goal, start_theta

    # Build the 5 perpendicular movers
    (human1y, targethuman1y, start_orientation_human1) = rand_edge_pair()
    (human2y, targethuman2y, start_orientation_human2) = rand_edge_pair()
    (human3y, targethuman3y, start_orientation_human3) = rand_edge_pair()
    (human4y, targethuman4y, start_orientation_human4) = rand_edge_pair()
    (human5y, targethuman5y, start_orientation_human5) = rand_edge_pair()

    human1x = float(x_positions[0])
    human2x = float(x_positions[1])
    human3x = float(x_positions[2])
    human4x = float(x_positions[3])
    human5x = float(x_positions[4])

    # Lateral (x) jitter kept minimal to remain “perpendicular”
    def jitter_x(x):
        return float(np.clip(x + np.random.uniform(-0.15, 0.15), x_min, x_max))

    human1x = jitter_x(human1x)
    human2x = jitter_x(human2x)
    human3x = jitter_x(human3x)
    human4x = jitter_x(human4x)
    human5x = jitter_x(human5x)

    # --- The remaining humans (6..10): keep as background or move them off-corridor ---
    # If you don't need them, you can keep them far or repurpose them.
    human6x = -30.38 + random_x
    human6y = 20.0
    start_orientation_human6 = 0.0
    targethuman6x = -0.0 + random_x
    targethuman6y = 20.0

    human7x = -29.26 + random_x
    human7y = 21.0
    start_orientation_human7 = 0.0
    targethuman7x = -70.0 + random_x
    targethuman7y = 21.0

    human8x = -28.56 + random_x
    human8y = 19.0
    start_orientation_human8 = 0.0
    targethuman8x = -70.0 + random_x
    targethuman8y = 22.0

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

    # --- Return all fields, preserving your original keys ---
    return {
        "mob_robot_startposx": float(mob_robot_startposx),
        "mob_robot_startposy": float(mob_robot_startposy),
        "mob_robot_start_orientation": float(mob_robot_start_orientation),
        "target_robot_x": float(target_robot_x),
        "target_robot_y": float(target_robot_y),
        "rad_angle": float(rad_angle),

        "human1x": human1x,
        "human1y": float(human1y),
        "start_orientation_human1": float(start_orientation_human1),
        "targethuman1x": human1x,
        "targethuman1y": float(targethuman1y),

        "human2x": human2x,
        "human2y": float(human2y),
        "start_orientation_human2": float(start_orientation_human2),
        "targethuman2x": human2x,
        "targethuman2y": float(targethuman2y),

        "human3x": human3x,
        "human3y": float(human3y),
        "start_orientation_human3": float(start_orientation_human3),
        "targethuman3x": human3x,
        "targethuman3y": float(targethuman3y),

        "human4x": human4x,
        "human4y": float(human4y),
        "start_orientation_human4": float(start_orientation_human4),
        "targethuman4x": human4x,
        "targethuman4y": float(targethuman4y),

        "human5x": human5x,
        "human5y": float(human5y),
        "start_orientation_human5": float(start_orientation_human5),
        "targethuman5x": human5x,
        "targethuman5y": float(targethuman5y),

        "human6x": float(human6x),
        "human6y": float(human6y),
        "start_orientation_human6": float(start_orientation_human6),
        "targethuman6x": float(targethuman6x),
        "targethuman6y": float(targethuman6y),

        "human7x": float(human7x),
        "human7y": float(human7y),
        "start_orientation_human7": float(start_orientation_human7),
        "targethuman7x": float(targethuman7x),
        "targethuman7y": float(targethuman7y),

        "human8x": float(human8x),
        "human8y": float(human8y),
        "start_orientation_human8": float(start_orientation_human8),
        "targethuman8x": float(targethuman8x),
        "targethuman8y": float(targethuman8y),

        "human9x": float(human9x),
        "human9y": float(human9y),
        "start_orientation_human9": float(start_orientation_human9),
        "targethuman9x": float(targethuman9x),
        "targethuman9y": float(targethuman9y),

        "human10x": float(human10x),
        "human10y": float(human10y),
        "start_orientation_human10": float(start_orientation_human10),
        "targethuman10x": float(targethuman10x),
        "targethuman10y": float(targethuman10y),
    }






