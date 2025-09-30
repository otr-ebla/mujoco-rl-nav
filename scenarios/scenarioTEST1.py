# import numpy as np
# import xml.etree.ElementTree as ET
# import gymnasium as gym 
# import mujoco
# import mujoco.viewer
# # Scenario 4: Corridoio lungo

# def scenarioTEST1():
#     random_x = np.random.uniform(-8, 8)
#     random_xr = np.random.uniform(-4.5, 4.5)
#     random_yr = np.random.uniform(-3.0, 3.0)
#     random_y = np.random.uniform(-3.0, 3.0)

#     randomy1 = np.random.uniform(-3.0, 3.0)
#     randomy2 = np.random.uniform(-3.0, 3.0)
#     randomy3 = np.random.uniform(-3.0, 3.0)
#     randomy4 = np.random.uniform(-3.0, 3.0)
#     randomy5 = np.random.uniform(-3.0, 3.0)

#     randomy6 = np.random.uniform(-3.0, 3.0)
#     randomy7 = np.random.uniform(-3.0, 3.0)
#     randomy8 = np.random.uniform(-3.0, 3.0)
#     randomy9 = np.random.uniform(-3.0, 3.0)
#     randomy10 = np.random.uniform(-3.0, 3.0)

#     random_angle = np.random.uniform(-45, 45)
#     rad_angle = np.deg2rad(random_angle)
    
#     mob_robot_startposx = 60.1 + random_xr
#     mob_robot_startposy = 19 + random_yr

#     in_rad_180 = np.deg2rad(180)
#     mob_robot_start_orientation = in_rad_180 + rad_angle
    
#     target_robot_x = 36.9 + random_xr
#     target_robot_y = 20 + random_yr



#     # humans
#     # red
#     human1x = 44.56 + random_x
#     human1y = 19.43 + randomy1
#     start_orientation_human1 = 0
#     targethuman1x = 71.1+ random_x
#     targethuman1y = 16.21 + randomy1

#     human2x = 41.0 + random_x
#     human2y = 20.5 + randomy2
#     start_orientation_human2 = 0
#     targethuman2x = 70.4+random_x
#     targethuman2y = 20.6 + randomy2

#     human3x = 40.61 + random_x
#     human3y = 18.8 + randomy3
#     start_orientation_human3 = 0.0
#     targethuman3x = 70.08 + random_x
#     targethuman3y = 18.8 + randomy3

#     # pink
#     human4x = 36.6 + random_x
#     human4y = 19.0 + randomy4
#     start_orientation_human4 = 0.0
#     targethuman4x = 70.1+random_x
#     targethuman4y = 16.0 + randomy4

#     human5x = 38.1 + random_x
#     human5y = 19.8 + randomy5
#     start_orientation_human5 = 0.0
#     targethuman5x = 59.0 + random_x
#     targethuman5y = 19.8 + randomy5

#     human6x = 30.38 + random_x
#     human6y = 20.0 + randomy6
#     start_orientation_human6 = 0.0
#     targethuman6x = 70.0 + random_x
#     targethuman6y = 20.0 + randomy6

#     human7x = 29.26 + random_x
#     human7y = 21.0 + randomy7
#     start_orientation_human7 = 0.0
#     targethuman7x = 70.0 + random_x
#     targethuman7y = 21.0 + randomy7

#     human8x = 28.56 + random_x
#     human8y = 19.0 + randomy8
#     start_orientation_human8 = 0.0
#     targethuman8x = 70.0 + random_x
#     targethuman8y = 22.0 + randomy8

#     human9x = 29.0 + random_x
#     human9y = 19.0 + randomy9
#     start_orientation_human9 = 0.0
#     targethuman9x = 70.0 + random_x
#     targethuman9y = 23.0 + randomy9

#     human10x = 24.6 + random_x
#     human10y = 19.0 + randomy10
#     start_orientation_human10 = 0.0
#     targethuman10x = 70.0 + random_x
#     targethuman10y = 24.0 + randomy10


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
#         "targethuman10y": targethuman10y,
#     }






import numpy as np
from env_config import N_HUMANS

# Scenario TEST1 (parallelo/contro-corrente) -- versione "spread" con distanza minima, 20 umani
def scenarioTEST1(seed=None):
    if seed is not None:
        np.random.seed(seed)

    # --- Corridoio Y (adatta se serve) ---
    Y_MIN, Y_MAX = 15, 24
    Y_MARGIN = 0.10
    y_low  = Y_MIN + Y_MARGIN
    y_high = Y_MAX - Y_MARGIN

    # --- Robot verso -x; umani verso +x ---
    random_xr = np.random.uniform(-4.5,  4.5)
    random_yr = np.random.uniform(-1.0,  1.0)
    random_angle = np.random.uniform(-180.0, 180.0)
    rad_angle = np.deg2rad(random_angle)

    mob_robot_startposx = 60.1 + random_xr
    mob_robot_startposy = float(np.clip(19.0 + random_yr, y_low, y_high))
    mob_robot_start_orientation = np.deg2rad(180.0) + rad_angle

    target_robot_x = 36.9 + random_xr
    target_robot_y = float(np.clip(20.0 + random_yr, y_low, y_high))

    # --- Umani: 20 in contro-corrente (+x), distribuiti "spread" ---
    n_hum = N_HUMANS-6

    # Finestra X di spawn (tra target e start del robot)
    x_min = target_robot_x + 0.5
    x_max = mob_robot_startposx - 8.0
    if x_max <= x_min + 1.0:
        x_max = x_min + 3.0  # fallback

    # Parametri di "spargimento": partiamo larghi, poi ci adattiamo se serve
    MIN_SEP_X_INIT = 1.6
    MIN_SEP_Y_INIT = 0.9

    def sample_with_min_sep_adaptive(k, x_lo, x_hi, y_lo, y_hi,
                                     min_sep_x_init, min_sep_y_init,
                                     steps=4, decay=0.8, max_tries=6000):
        """Tenta con soglie alte e le riduce progressivamente se lo spazio non basta."""
        min_x = min_sep_x_init
        min_y = min_sep_y_init
        for _ in range(steps):
            pts, tries = [], 0
            while len(pts) < k and tries < max_tries:
                x = np.random.uniform(x_lo, x_hi)
                y = np.random.uniform(y_lo, y_hi)
                ok = True
                for (px, py) in pts:
                    if abs(x - px) < min_x and abs(y - py) < min_y:
                        ok = False
                        break
                if ok:
                    pts.append((float(x), float(y)))
                tries += 1
            if len(pts) >= k:
                return pts[:k]
            # non abbastanza punti: allentiamo i vincoli e riproviamo
            min_x *= decay
            min_y *= decay

        # Fallback: griglia sparsa se ancora insufficienti
        nx = max(1, int((x_hi - x_lo) / max(min_x, 0.4)))
        ny = max(1, int((y_hi - y_lo) / max(min_y, 0.25)))
        grid = []
        for i in range(nx):
            for j in range(ny):
                gx = x_lo + (i + 0.5) * (x_hi - x_lo) / max(nx, 1)
                gy = y_lo + (j + 0.5) * (y_hi - y_lo) / max(ny, 1)
                gx += np.random.uniform(-0.15, 0.15)
                gy += np.random.uniform(-0.08, 0.08)
                grid.append((float(np.clip(gx, x_lo, x_hi)),
                             float(np.clip(gy, y_lo, y_hi))))
        np.random.shuffle(grid)
        return grid[:k]

    # Spawn "spread" nel rettangolo [x_min,x_max] × [y_low,y_high]
    starts = sample_with_min_sep_adaptive(
        n_hum, x_min, x_max, y_low, y_high, MIN_SEP_X_INIT, MIN_SEP_Y_INIT
    )

    humans = []
    for (x0, y0) in starts:
        # Heading ~ 0° con piccolo rumore → paralleli al corridoio verso +x
        heading = float(np.random.uniform(-3.5, 3.5))
        # Target x oltre lo start del robot (così incrociano di sicuro)
        x_goal = float(mob_robot_startposx + np.random.uniform(8.0, 18.0))
        # Deriva laterale minima, clamp nel corridoio
        y_goal = float(np.clip(y0 + np.random.uniform(-0.15, 0.15), y_low, y_high))
        humans.append((float(x0), float(y0), heading, x_goal, y_goal))

    # Mischia per non avere ordine spaziale fisso
    np.random.shuffle(humans)

    # --- Padding/Truncate a N_HUMANS ---
    OFF_X = -100     # "fuori mappa"
    OFF_Y = 0.0
    # Se meno del necessario, aggiungi dummy fuori mappa
    while len(humans) < N_HUMANS:
        humans.append((OFF_X, OFF_Y, 0.0, OFF_X, OFF_Y))
    # Se per caso sono di più, taglia
    if len(humans) > N_HUMANS:
        humans = humans[:N_HUMANS]

    # --- Costruisci il dict di ritorno con chiavi human1..human20 ---
    result = {
        "mob_robot_startposx": float(mob_robot_startposx),
        "mob_robot_startposy": float(mob_robot_startposy),
        "mob_robot_start_orientation": float(mob_robot_start_orientation),
        "target_robot_x": float(target_robot_x),
        "target_robot_y": float(target_robot_y),
        "rad_angle": float(rad_angle),
    }

    for i, (hx, hy, htheta, tx, ty) in enumerate(humans, start=1):
        result[f"human{i}x"] = float(hx)
        result[f"human{i}y"] = float(hy)
        result[f"start_orientation_human{i}"] = float(htheta)
        result[f"targethuman{i}x"] = float(tx)
        result[f"targethuman{i}y"] = float(ty)

    return result
