import numpy as np

# Scenario 6: DUE PORTE LATERALI — 1 umano attivo + padding fino a 20 (off-map)
def scenario6():
    rand_dim = 2.0
    max_x = 0.5
    random_x = np.random.uniform(-rand_dim, rand_dim)
    random_y = np.random.uniform(-rand_dim, rand_dim)
    rand_targ_x = np.random.uniform(-max_x, max_x)
    random_angle = np.random.uniform(-179, 179)
    rad_angle = np.deg2rad(random_angle)

    # Robot (come nel tuo file)
    mob_robot_startposx = 55.56 + rand_targ_x
    mob_robot_startposy = 16.67
    mob_robot_start_orientation = np.deg2rad(-90) + rad_angle
    target_robot_x = 55.1 + rand_targ_x
    target_robot_y = 5.64

    # --- 1 UMANO ATTIVO (come nel tuo file, con random_x/random_y) ---
    human1x = 48.57 + random_x
    human1y = 9.2 + random_y
    start_orientation_human1 = 0.0
    targethuman1x = 60.55 + random_x
    targethuman1y = 10.2 + random_y

    # --- TUTTI GLI ALTRI FUORI MAPPA ---
    N_HUMANS = 20
    OFF_X = -1e6
    OFF_Y = 0.0
    OFF_TH = 0.0

    # Costruisci lista: primo attivo, poi padding
    humans = [(human1x, human1y, start_orientation_human1, targethuman1x, targethuman1y)]
    # piccoli offset Y opzionali per chiavi distinte
    for i in range(2, N_HUMANS + 1):
        dy = (i - 10) * 0.01
        humans.append((OFF_X, OFF_Y + dy, OFF_TH, OFF_X, OFF_Y + dy))

    # --- RITORNO CON CHIAVI human1..human20 ---
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
