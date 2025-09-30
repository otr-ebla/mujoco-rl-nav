import numpy as np

# Scenario5: 5 verticali con tratto minimo 8 m + padding a 20
def scenario5():
    # --- Random base ---
    random_x = np.random.uniform(-4.5, 4.5)
    random_y  = np.random.uniform(-2, 2)
    random_y2 = np.random.uniform(-2, 2)
    random_angle = np.random.uniform(-179, 179)
    rad_angle = np.deg2rad(random_angle)

    # --- Robot ---
    mob_robot_startposx = 40.2
    mob_robot_startposy = -13.46
    mob_robot_start_orientation = np.deg2rad(90) + rad_angle
    target_robot_x = 40.5
    target_robot_y = 10 + random_y

    # ========= Corridoio e tavoli =========
    # Intervalli liberi lungo Y (adatta se cambi mappa):
    FREE_SPANS_Y = [
        (-10.8, -3.6),
        (-5.0,   4.4),
        ( 2.8,   7.9),
        ( 3.2,  12.3),
    ]
    BASE_LANES_X = [39.1, 39.5, 40.43, 40.5]
    LANE_X_JITTER = 0.12

    N_ACTIVE = 5
    N_HUMANS = 20
    MIN_SEP_Y   = 2   # separazione minima tra pedoni nello stesso span
    MIN_TRAVEL  = 4    # *** nuovo: tragitto minimo 8 m ***
    EDGE_MARGIN = 0.10    # margine di sicurezza dagli estremi dello span
    GOAL_JITTER = 0.25    # jitter sul goal

    def pick_lane_x():
        x = np.random.choice(BASE_LANES_X)
        return float(x + np.random.uniform(-LANE_X_JITTER, LANE_X_JITTER))

    def sample_y_in_span(span, used, min_sep):
        y0, y1 = span
        for _ in range(2000):
            y = np.random.uniform(y0 + EDGE_MARGIN, y1 - EDGE_MARGIN)
            if all(abs(y - uy) >= min_sep for uy in used):
                return float(y)
        # fallback: griglia
        grid = np.linspace(y0 + 0.2, y1 - 0.2, max(2, int((y1 - y0) / max(0.4, min_sep))))
        np.random.shuffle(grid)
        for y in grid:
            if all(abs(y - uy) >= min_sep for uy in used):
                return float(y)
        return float((y0 + y1) / 2.0)

    # Filtra gli span che permettono almeno 8 m in totale
    eligible_spans = [(i, s) for i, s in enumerate(FREE_SPANS_Y) if (s[1] - s[0]) >= (MIN_TRAVEL + 2*EDGE_MARGIN)]
    if not eligible_spans:
        # se nessuno span è sufficiente, usa i più lunghi e porteremo il goal al max disponibile
        eligible_spans = list(enumerate(FREE_SPANS_Y))

    humans = []
    span_used_y = {idx: [] for idx, _ in eligible_spans}

    for _ in range(N_ACTIVE):
        # ricampiona finché troviamo start+direzione che garantiscano >= 8 m
        for _try in range(3000):
            span_idx, span = eligible_spans[np.random.randint(0, len(eligible_spans))]
            y0, y1 = span
            # scegli x-corsia
            lane_x = pick_lane_x()
            # scegli y_start
            y_start = sample_y_in_span(span, span_used_y[span_idx], MIN_SEP_Y)

            # spazio disponibile verso l'alto e verso il basso
            up_room   = (y1 - EDGE_MARGIN) - y_start
            down_room = y_start - (y0 + EDGE_MARGIN)

            choices = []
            if up_room >= MIN_TRAVEL:   choices.append("up")
            if down_room >= MIN_TRAVEL: choices.append("down")

            if choices:
                go_dir = np.random.choice(choices)
                if go_dir == "up":
                    # target almeno 8 m sopra, ma non oltre il bordo
                    y_goal = y_start + np.random.uniform(MIN_TRAVEL, min(MIN_TRAVEL + 2.5, up_room))
                    start_theta = 90.0
                else:
                    y_goal = y_start - np.random.uniform(MIN_TRAVEL, min(MIN_TRAVEL + 2.5, down_room))
                    start_theta = -90.0

                y_goal = float(np.clip(y_goal + np.random.uniform(-GOAL_JITTER, GOAL_JITTER),
                                       y0 + EDGE_MARGIN, y1 - EDGE_MARGIN))
                span_used_y[span_idx].append(y_start)

                humans.append((lane_x, float(y_start), start_theta, lane_x, y_goal))
                break
        else:
            # fallback improbabile: se non riusciamo, metti qualcuno off-map
            humans.append((-1e6, 0.0, 0.0, -1e6, 0.0))

    # Padding fino a 20
    OFF_X = -1e6
    OFF_Y = 0.0
    OFF_TH = 0.0
    while len(humans) < N_HUMANS:
        i = len(humans) + 1
        dy = (i - 10) * 0.01
        humans.append((OFF_X, OFF_Y + dy, OFF_TH, OFF_X, OFF_Y + dy))

    # Build dict
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
        result[f"start_orientation_human{i}"] = float(htheta)  # gradi ±90
        result[f"targethuman{i}x"] = float(tx)
        result[f"targethuman{i}y"] = float(ty)
    return result
