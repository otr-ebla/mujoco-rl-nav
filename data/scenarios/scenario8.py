import numpy as np

# Scenario 8: in fondo all'ambiente attraversa la porta
# -> 8 umani attivi + padding fino a 20 (off-map)
def scenario8():
    # --- Random base ---
    random_x = np.random.uniform(-4.5, 4.5)
    random_y = np.random.uniform(-2.5, 2.5)

    # >>> d'ora in poi: random_angle in [-179, 179]
    random_angle = np.random.uniform(-179, 179)
    rad_angle = np.deg2rad(random_angle)

    # --- Robot ---
    mob_robot_startposx = 94.0
    mob_robot_startposy = 14.5 + random_y
    mob_robot_start_orientation = np.deg2rad(-90.0) + rad_angle

    target_robot_x = 93.88
    target_robot_y = -3.4 + random_y

    # --- 8 UMANI ATTIVI (coerenti con l'area vicino alla porta) ---
    # Mantengo i primi 3 del tuo file e ne aggiungo altri 5 con piccola randomizzazione.
    human1x, human1y = 93.9, 3.36
    start_orientation_human1 = 90.0
    targethuman1x, targethuman1y = 93.9, 10.22 + random_y

    human2x, human2y = 95.98, 2.2
    start_orientation_human2 = 180.0
    targethuman2x, targethuman2y = 91.4, 2.2

    human3x, human3y = 91.34, 11.79
    start_orientation_human3 = 0.0
    targethuman3x, targethuman3y = 96.13, 11.79



    # --- Padding: umani 9..20 fuori mappa ---
    N_HUMANS = 20
    OFF_X = -1e6
    OFF_Y = 0.0
    OFF_TH = 0.0

    humans = [
        (human1x, human1y, start_orientation_human1, targethuman1x, targethuman1y),
        (human2x, human2y, start_orientation_human2, targethuman2x, targethuman2y),
        (human3x, human3y, start_orientation_human3, targethuman3x, targethuman3y),
    ]

    # padding fino a 20
    for i in range(4, N_HUMANS + 1):
        dy = (i - 10) * 0.01  # piccolo offset distintivo, opzionale
        humans.append((OFF_X, OFF_Y + dy, OFF_TH, OFF_X, OFF_Y + dy))

    # --- Return dict con tutte le chiavi richieste ---
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
