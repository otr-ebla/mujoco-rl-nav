# import numpy as np

# # Scenario 1: Il robot attraversa il corridoio centrale iniziale e raggiunge il target al fondo davanti alla parete
# #             Gli umani si incrociano lì davanti

# def scenarioTEST3():
#     random_x = np.random.uniform(0, 6)
#     random_y = np.random.uniform(-4, 4)

#     random_angle = np.random.uniform(-45, 45)
#     rad_angle = np.deg2rad(random_angle)

#     mob_robot_startposx = 17.26 + random_x
#     mob_robot_startposy = 0
#     init_angle = 180
#     init_rad = np.deg2rad(init_angle)
#     mob_robot_start_orientation = init_rad + rad_angle
#     target_robot_x = 2.26
#     target_robot_y = 0 + random_y

#     human1x = 14.36
#     human1y = -3.36
#     start_orientation_human1 = 90
#     targethuman1x = 14.36
#     targethuman1y = 2.73

#     human2x = 12.5
#     human2y = 2.73
#     start_orientation_human2 = -90
#     targethuman2x = 12.5
#     targethuman2y = -3.36

#     human3x = 9
#     human3y = -3.36
#     start_orientation_human3 = 90.0
#     targethuman3x = 9
#     targethuman3y = 2.73

#     human4x = 6.5
#     human4y = 2.73
#     start_orientation_human4 = -90.0
#     targethuman4x = 6.5
#     targethuman4y = -3.36

#     human5x = 3.85
#     human5y = -3.36
#     start_orientation_human5 = 90.0
#     targethuman5x = 3.85
#     targethuman5y = 2.73



#     human6x = 10.5
#     human6y = 2.73
#     start_orientation_human6 = 90.0
#     targethuman6x = 10.5
#     targethuman6y = -3.36

#     human7x = 5
#     human7y = 2.73
#     start_orientation_human7 = -90.0
#     targethuman7x = 5
#     targethuman7y = -3.36

#     human8x = 28.56 + random_x
#     human8y = 9.29 + random_y 
#     start_orientation_human8 = -90
#     targethuman8x = 28.56 + random_x
#     targethuman8y = -4.9 + random_y

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

# Scenario TEST3: 7 umani che attraversano perpendicolarmente il corridoio
def scenarioTEST3(seed=None):
    if seed is not None:
        np.random.seed(seed)

    # --- Geometria corridoio (dai tuoi valori) ---
    Y_MIN, Y_MAX = -4.5, 4.0
    Y_MARGIN = 0.12
    y_low  = Y_MIN + Y_MARGIN
    y_high = Y_MAX - Y_MARGIN

    # --- Start/target robot (coerente col tuo file) ---
    random_x = np.random.uniform(0, 6.0)
    random_y = np.random.uniform(-4.0, 4.0)
    random_angle = np.random.uniform(-45.0, 45.0)
    rad_angle = np.deg2rad(random_angle)

    mob_robot_startposx = 17.26 + random_x
    mob_robot_startposy = 0.0
    init_angle = 180.0
    init_rad = np.deg2rad(init_angle)
    mob_robot_start_orientation = init_rad + rad_angle

    target_robot_x = 2.26
    target_robot_y = float(np.clip(0.0 + random_y, y_low, y_high))

    # --- 7 "crossers" in X tra target e start, con spaziatura minima ---
    n_crossers = 7
    x_min = target_robot_x + 1.0
    x_max = mob_robot_startposx - 2.0
    if x_max <= x_min + 1.0:
        x_max = x_min + 3.0  # fallback

    def sample_sorted_positions(n, lo, hi, min_sep):
        xs, tries = [], 0
        while len(xs) < n and tries < 2000:
            c = np.random.uniform(lo, hi)
            if all(abs(c - v) >= min_sep for v in xs):
                xs.append(c)
            tries += 1
        if len(xs) < n:
            xs = list(np.linspace(lo, hi, n))
        xs.sort()
        return xs

    min_sep = max(0.8, 0.12 * max(x_max - x_min, 1.0))
    x_positions = sample_sorted_positions(n_crossers, x_min, x_max, min_sep)

    # --- helper: lato di partenza (su/giù) + jitter e heading coerente ---
    def rand_edge_pair():
        start_top = np.random.rand() < 0.5
        jitter = np.random.uniform(0.0, 0.25)
        if start_top:
            y_start = y_high - jitter
            y_goal  = y_low  + jitter
            start_theta = -90.0  # verso il basso
        else:
            y_start = y_low  + jitter
            y_goal  = y_high - jitter
            start_theta = 90.0   # verso l'alto
        return y_start, y_goal, start_theta

    def jitter_x(x):
        return float(np.clip(x + np.random.uniform(-0.12, 0.12), x_min, x_max))

    # --- Assegna i 7 pedoni perpendicolari ---
    human_xy = []
    for i in range(n_crossers):
        y_start, y_goal, theta = rand_edge_pair()
        x = jitter_x(float(x_positions[i]))
        human_xy.append((x, y_start, x, y_goal, theta))

    # Scompatta in ordine 1..7
    (human1x, human1y, targethuman1x, targethuman1y, start_orientation_human1) = human_xy[0]
    (human2x, human2y, targethuman2x, targethuman2y, start_orientation_human2) = human_xy[1]
    (human3x, human3y, targethuman3x, targethuman3y, start_orientation_human3) = human_xy[2]
    (human4x, human4y, targethuman4x, targethuman4y, start_orientation_human4) = human_xy[3]
    (human5x, human5y, targethuman5x, targethuman5y, start_orientation_human5) = human_xy[4]
    (human6x, human6y, targethuman6x, targethuman6y, start_orientation_human6) = human_xy[5]
    (human7x, human7y, targethuman7x, targethuman7y, start_orientation_human7) = human_xy[6]

    # --- Gli altri (8..10) li lasciamo fuori/di sfondo, come nel tuo file ---
    human8x = 28.56 + random_x
    human8y = 9.29 + random_y
    start_orientation_human8 = -90.0
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

    # --- Ritorna con le stesse chiavi del tuo scenario originale ---
    return {
        "mob_robot_startposx": float(mob_robot_startposx),
        "mob_robot_startposy": float(mob_robot_startposy),
        "mob_robot_start_orientation": float(mob_robot_start_orientation),
        "target_robot_x": float(target_robot_x),
        "target_robot_y": float(target_robot_y),
        "rad_angle": float(rad_angle),

        "human1x": float(human1x),
        "human1y": float(human1y),
        "start_orientation_human1": float(start_orientation_human1),
        "targethuman1x": float(targethuman1x),
        "targethuman1y": float(targethuman1y),

        "human2x": float(human2x),
        "human2y": float(human2y),
        "start_orientation_human2": float(start_orientation_human2),
        "targethuman2x": float(targethuman2x),
        "targethuman2y": float(targethuman2y),

        "human3x": float(human3x),
        "human3y": float(human3y),
        "start_orientation_human3": float(start_orientation_human3),
        "targethuman3x": float(targethuman3x),
        "targethuman3y": float(targethuman3y),

        "human4x": float(human4x),
        "human4y": float(human4y),
        "start_orientation_human4": float(start_orientation_human4),
        "targethuman4x": float(targethuman4x),
        "targethuman4y": float(targethuman4y),

        "human5x": float(human5x),
        "human5y": float(human5y),
        "start_orientation_human5": float(start_orientation_human5),
        "targethuman5x": float(targethuman5x),
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
