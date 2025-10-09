import numpy as np

# Scenario 7: robot tra le colonne -> 5 umani attivi + padding fino a 20 (off-map)
def scenario7():
    # --- Random base ---
    random_x  = np.random.uniform(-4.5, 4.5)
    random_y2 = np.random.uniform(-3.0, 3.0)
    random_y  = np.random.uniform(-4.5, 4.5)

    # >>> da ora in poi: random_angle in [-179, 179]
    random_angle = np.random.uniform(-179, 179)
    rad_angle = np.deg2rad(random_angle)

    # --- Robot ---
    mob_robot_startposx = 73.0
    mob_robot_startposy = -10.76 + random_y
    in_rad_90 = np.deg2rad(90.0)
    mob_robot_start_orientation = in_rad_90 + rad_angle

    target_robot_x = 73.0
    target_robot_y = 9.08 + random_y2

    # --- 5 UMANI ATTIVI (coerenti con il tuo scenario originale, con un filo di random) ---
    # 1: attraversa in x
    human1x, human1y = 66.5, 0.0 + random_y
    start_orientation_human1 = 0.0
    targethuman1x, targethuman1y = 73.2, 0.0

    # 2: verticale dall'alto verso il basso
    human2x, human2y = 67.1, 9.2
    start_orientation_human2 = -90.0
    targethuman2x, targethuman2y = 67.1, -9.8

    # 3: verticale dal basso verso l'alto
    human3x, human3y = 72.5+ np.random.uniform(-1.5, 1.5), -9.3
    start_orientation_human3 = 90.0
    targethuman3x, targethuman3y = 72.5 + np.random.uniform(-1.5, 1.5), 8.48

    # 4: verticale con leggera variazione
    human4x, human4y = 69.4, -6.0 + np.random.uniform(-1.0, 1.0)
    start_orientation_human4 = 90.0
    targethuman4x, targethuman4y = 69.4, 7.5 + np.random.uniform(-0.8, 0.8)

    # 5: orizzontale leggera diagonale
    human5x, human5y = 68.2, 1.4 + np.random.uniform(-1.2, 1.2)
    start_orientation_human5 = np.random.choice([-10.0, 10.0])  # ~orizzontale
    targethuman5x, targethuman5y = 73.9, human5y + np.random.uniform(-0.6, 0.6)

    # --- Padding: altri 15 umani fuori mappa ---
    N_HUMANS = 20
    OFF_X = -1e6
    OFF_Y = 0.0
    OFF_TH = 0.0

    humans = [
        (human1x, human1y, start_orientation_human1, targethuman1x, targethuman1y),
        (human2x, human2y, start_orientation_human2, targethuman2x, targethuman2y),
        (human3x, human3y, start_orientation_human3, targethuman3x, targethuman3y),
        (human4x, human4y, start_orientation_human4, targethuman4x, targethuman4y),
        (human5x, human5y, start_orientation_human5, targethuman5x, targethuman5y),
    ]

    # aggiungi dummy fino a 20
    for i in range(6, N_HUMANS + 1):
        dy = (i - 10) * 0.01  # piccolo offset distintivo opzionale
        humans.append((OFF_X, OFF_Y + dy, OFF_TH, OFF_X, OFF_Y + dy))

    # --- Build return dict con human1..human20 ---
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
