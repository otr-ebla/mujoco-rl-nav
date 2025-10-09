# import numpy as np
# import xml.etree.ElementTree as ET
# import gymnasium as gym 
# import mujoco
# import mujoco.viewer

# # Scenario 1: Il robot attraversa il corridoio centrale iniziale e raggiunge il target al fondo davanti alla parete
# #             Gli umani si incrociano lì davanti

# def scenarioTEST2():
#     delta_random = 3
#     delta_target_random_y = np.random.uniform(-5, 5)
#     delta_target_random_x = np.random.uniform(-3, 0)


#     random_x = np.random.uniform(-delta_random, delta_random)
#     random_y = np.random.uniform(-delta_random, delta_random)
#     random_angle = np.random.uniform(-45, 45)
#     rad_angle = np.deg2rad(random_angle)
#     mob_robot_startposx = 16.85 + random_x
#     mob_robot_startposy = 0

#     mob_robot_start_orientation = 0 + rad_angle
    
#     target_robot_x = 32.8 + delta_target_random_x
#     target_robot_y = 0.0 + delta_target_random_y
    
#     human1x = 27.43 + random_x
#     human1y = -8.7 + random_y
#     start_orientation_human1 = 0.0
#     targethuman1x = 27.43
#     targethuman1y = 7.63 + random_y

#     human2x = 21.67
#     human2y = -9 + random_y
#     start_orientation_human2 = -20
#     targethuman2x = 32
#     targethuman2y = 6.74 + random_y

#     human3x = 23
#     human3y = 6.74 + random_y
#     start_orientation_human3 = -150
#     targethuman3x = 32.4
#     targethuman3y = -6.8 + random_y

#     human4x = 32.3
#     human4y = -2.6 + random_y
#     start_orientation_human4 = 90.0
#     targethuman4x = 16.26
#     targethuman4y = -2.6 + random_y

#     human5x = 16.26 + random_x
#     human5y = 2.77
#     start_orientation_human5 = -90.0
#     targethuman5x = 29.45 + random_x
#     targethuman5y = 2.77 + random_y

#     human6x = 30.23 + random_x
#     human6y = -6.0 + random_y
#     start_orientation_human6 = 90.0
#     targethuman6x = 21.26 + random_x
#     targethuman6y = 4.18

#     human7x = 30.26 + random_x
#     human7y = 9.0 + random_y 
#     start_orientation_human7 = -90.0
#     targethuman7x = 21.26 + random_x
#     targethuman7y = -3.3 

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

# Scenario TEST2: Incrocio obliquo con flussi diagonali che si intersecano (random)
import numpy as np

# Scenario TEST2: Incrocio obliquo, umani MENO raggruppati (spread con Poisson sampling)
def scenarioTEST2(seed=None):
    if seed is not None:
        np.random.seed(seed)

    # --- Parametri di "spaziatura" (aumenta per meno raggruppamento) ---
    MIN_SEP_X = 1.4   # distanza minima tra spawn in X
    MIN_SEP_Y = 0.9   # distanza minima tra spawn in Y

    # --- Random di base per robot (come prima) ---
    delta_random = 3.0
    random_x = np.random.uniform(-delta_random, delta_random)
    random_y = np.random.uniform(-delta_random, delta_random)
    random_angle = np.random.uniform(-180, 180)
    rad_angle = np.deg2rad(random_angle)

    mob_robot_startposx = 16.85 + random_x
    mob_robot_startposy = 0.0
    mob_robot_start_orientation = 0.0 + rad_angle

    delta_target_random_y = np.random.uniform(-5.0, 5.0)
    delta_target_random_x = np.random.uniform(-3.0, 0.0)
    target_robot_x = 32.8 + delta_target_random_x
    target_robot_y = 0.0 + delta_target_random_y

    # --- Rettangolo d'incrocio (adatta ai tuoi limiti reali) ---
    X_MIN, X_MAX = 20.5, 31.5
    Y_MIN, Y_MAX = -11.0, 11.0
    XM, YM = 0.4, 0.4

    def clamp_xy(x, y):
        return float(np.clip(x, X_MIN+XM, X_MAX-XM)), float(np.clip(y, Y_MIN+YM, Y_MAX-YM))

    # Flussi diagonali (angoli)
    flows = {
        "NE":  lambda: np.random.uniform(25.0, 60.0),
        "NW":  lambda: np.random.uniform(120.0, 155.0),
        "SE":  lambda: np.random.uniform(-60.0, -25.0),
        "SW":  lambda: np.random.uniform(-155.0, -120.0),
    }

    n = 10
    # 3–4 flussi attivi per distribuire meglio la gente
    n_active = np.random.randint(3, 5)  # 3 o 4
    active_flows = np.random.choice(list(flows.keys()), size=n_active, replace=False)

    # Ripartizione quasi uniforme (meno clusteroni): differenze al più ±1
    base = n // n_active
    sizes = np.array([base] * n_active)
    for i in np.random.permutation(n_active)[: n - sizes.sum()]:
        sizes[i] += 1

    # Helper per campionare punti con separazione minima (Poisson "dart throwing" semplice)
    def sample_with_min_sep(k, x_lo, x_hi, y_lo, y_hi, min_sep_x, min_sep_y, max_tries=8000):
        pts = []
        tries = 0
        while len(pts) < k and tries < max_tries:
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            ok = True
            for (px, py) in pts:
                if abs(x - px) < min_sep_x and abs(y - py) < min_sep_y:
                    ok = False
                    break
            if ok:
                pts.append((float(x), float(y)))
            tries += 1
        # fallback: griglia se non basta lo spazio
        if len(pts) < k:
            nx = max(1, int((x_hi - x_lo) / min_sep_x))
            ny = max(1, int((y_hi - y_lo) / min_sep_y))
            grid = []
            for i in range(nx):
                for j in range(ny):
                    gx = x_lo + (i + 0.5) * (x_hi - x_lo) / nx
                    gy = y_lo + (j + 0.5) * (y_hi - y_lo) / ny
                    grid.append((gx, gy))
            np.random.shuffle(grid)
            for g in grid:
                if len(pts) < k:
                    pts.append((float(g[0]), float(g[1])))
        return pts[:k]

    # Funzione per scegliere rettangoli di spawn/goal lungo i bordi coerenti col flusso
    def edge_rect(edge, span_frac=0.85, thick=0.8):
        # span_frac: percentuale di estensione lungo il bordo; thick: spessore verso l'interno
        span_x = (X_MAX - X_MIN) * span_frac
        span_y = (Y_MAX - Y_MIN) * span_frac
        if edge == "left":
            x_lo, x_hi = X_MIN+XM, X_MIN+XM+thick
            y_mid = 0.5 * (Y_MIN+YM + Y_MAX-YM)
            y_lo, y_hi = y_mid - span_y/2, y_mid + span_y/2
        elif edge == "right":
            x_lo, x_hi = X_MAX-XM-thick, X_MAX-XM
            y_mid = 0.5 * (Y_MIN+YM + Y_MAX-YM)
            y_lo, y_hi = y_mid - span_y/2, y_mid + span_y/2
        elif edge == "bottom":
            y_lo, y_hi = Y_MIN+YM, Y_MIN+YM+thick
            x_mid = 0.5 * (X_MIN+XM + X_MAX-XM)
            x_lo, x_hi = x_mid - span_x/2, x_mid + span_x/2
        else:  # "top"
            y_lo, y_hi = Y_MAX-YM-thick, Y_MAX-YM
            x_mid = 0.5 * (X_MIN+XM + X_MAX-XM)
            x_lo, x_hi = x_mid - span_x/2, x_mid + span_x/2
        return x_lo, x_hi, y_lo, y_hi

    humans = []
    for flow_name, k in zip(active_flows, sizes):
        heading_sampler = flows[flow_name]
        theta0 = heading_sampler()

        # start/goal edges coerenti con il flusso (come prima)
        if flow_name == "NE":
            start_edge = np.random.choice(["left", "bottom"])
            goal_edge  = np.random.choice(["right", "top"])
        elif flow_name == "NW":
            start_edge = np.random.choice(["right", "bottom"])
            goal_edge  = np.random.choice(["left", "top"])
        elif flow_name == "SE":
            start_edge = np.random.choice(["left", "top"])
            goal_edge  = np.random.choice(["right", "bottom"])
        else:  # SW
            start_edge = np.random.choice(["right", "top"])
            goal_edge  = np.random.choice(["left", "bottom"])

        # rettangoli più larghi lungo il bordo → persone già "sparse" sul bordo
        sx_lo, sx_hi, sy_lo, sy_hi = edge_rect(start_edge, span_frac=0.9, thick=1.2)
        gx_lo, gx_hi, gy_lo, gy_hi = edge_rect(goal_edge,  span_frac=0.9, thick=1.2)

        # campiona k start con separazione minima
        starts = sample_with_min_sep(k, sx_lo, sx_hi, sy_lo, sy_hi, MIN_SEP_X, MIN_SEP_Y)

        # per ogni start, campiona un goal indipendente con separazione (meno importante nel goal)
        goals  = sample_with_min_sep(k, gx_lo, gx_hi, gy_lo, gy_hi, MIN_SEP_X*0.6, MIN_SEP_Y*0.6)

        for (x0, y0), (xg, yg) in zip(starts, goals):
            x0, y0 = clamp_xy(x0, y0)
            xg, yg = clamp_xy(xg, yg)
            heading = float(theta0 + np.random.uniform(-6.0, 6.0))
            humans.append((x0, y0, heading, xg, yg))

    # Normalizza a n=10
    if len(humans) > n:
        humans = humans[:n]
    while len(humans) < n:
        humans.append(humans[-1])

    np.random.shuffle(humans)

    # Mappa sui 10 slot con le stesse chiavi del tuo file
    (human1x,  human1y,  start_orientation_human1,  targethuman1x,  targethuman1y)  = humans[0]
    (human2x,  human2y,  start_orientation_human2,  targethuman2x,  targethuman2y)  = humans[1]
    (human3x,  human3y,  start_orientation_human3,  targethuman3x,  targethuman3y)  = humans[2]
    (human4x,  human4y,  start_orientation_human4,  targethuman4x,  targethuman4y)  = humans[3]
    (human5x,  human5y,  start_orientation_human5,  targethuman5x,  targethuman5y)  = humans[4]
    (human6x,  human6y,  start_orientation_human6,  targethuman6x,  targethuman6y)  = humans[5]
    (human7x,  human7y,  start_orientation_human7,  targethuman7x,  targethuman7y)  = humans[6]
    (human8x,  human8y,  start_orientation_human8,  targethuman8x,  targethuman8y)  = humans[7]
    (human9x,  human9y,  start_orientation_human9,  targethuman9x,  targethuman9y)  = humans[8]
    (human10x, human10y, start_orientation_human10, targethuman10x, targethuman10y) = humans[9]

    return {
        "mob_robot_startposx": float(mob_robot_startposx),
        "mob_robot_startposy": float(mob_robot_startposy),
        "mob_robot_start_orientation": float(mob_robot_start_orientation),
        "target_robot_x": float(target_robot_x),
        "target_robot_y": float(target_robot_y),
        "rad_angle": float(rad_angle),

        "human1x": human1x, "human1y": human1y,
        "start_orientation_human1": start_orientation_human1,
        "targethuman1x": targethuman1x, "targethuman1y": targethuman1y,

        "human2x": human2x, "human2y": human2y,
        "start_orientation_human2": start_orientation_human2,
        "targethuman2x": targethuman2x, "targethuman2y": targethuman2y,

        "human3x": human3x, "human3y": human3y,
        "start_orientation_human3": start_orientation_human3,
        "targethuman3x": targethuman3x, "targethuman3y": targethuman3y,

        "human4x": human4x, "human4y": human4y,
        "start_orientation_human4": start_orientation_human4,
        "targethuman4x": targethuman4x, "targethuman4y": targethuman4y,

        "human5x": human5x, "human5y": human5y,
        "start_orientation_human5": start_orientation_human5,
        "targethuman5x": targethuman5x, "targethuman5y": targethuman5y,

        "human6x": human6x, "human6y": human6y,
        "start_orientation_human6": start_orientation_human6,
        "targethuman6x": targethuman6x, "targethuman6y": targethuman6y,

        "human7x": human7x, "human7y": human7y,
        "start_orientation_human7": start_orientation_human7,
        "targethuman7x": targethuman7x, "targethuman7y": targethuman7y,

        "human8x": human8x, "human8y": human8y,
        "start_orientation_human8": start_orientation_human8,
        "targethuman8x": targethuman8x, "targethuman8y": targethuman8y,

        "human9x": human9x, "human9y": human9y,
        "start_orientation_human9": start_orientation_human9,
        "targethuman9x": targethuman9x, "targethuman9y": targethuman9y,

        "human10x": human10x, "human10y": human10y,
        "start_orientation_human10": start_orientation_human10,
        "targethuman10x": targethuman10x, "targethuman10y": targethuman10y,
    }






