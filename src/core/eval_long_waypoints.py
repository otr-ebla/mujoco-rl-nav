# src/core/eval_long_waypoints.py
import numpy as np
import jax.numpy as jnp
import mujoco

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from lightHAMRRLN import light_hamrrln

def set_endless_humans_from_pairs(env, pairs):
    """
    pairs: list of ((x1,y1), (x2,y2)).
    Humans ping-pong forever between p1 and p2 using env._update_human_goals().
    """
    assert env.n_humans == len(pairs), f"n_humans={env.n_humans}, pairs={len(pairs)}"

    human_goals = np.zeros((env.n_humans, 2, 2), dtype=np.float32)
    for i, ((x1, y1), (x2, y2)) in enumerate(pairs):
        human_goals[i, 0, :] = [x1, y1]
        human_goals[i, 1, :] = [x2, y2]

        # init mocap at p1 and heading to p2
        th = float(np.arctan2((y2 - y1), (x2 - x1)))
        mocap_id = int(env.human_mocap_ids[i])
        env.data.mocap_pos[mocap_id]  = [x1, y1, 0.0]
        env.data.mocap_quat[mocap_id] = [np.cos(th/2.0), 0.0, 0.0, np.sin(th/2.0)]
        env.humans_state_numpy[i, :2] = [x1, y1]
        env.humans_state_numpy[i, 2:4] = 0.0
        env.humans_state_numpy[i, 4] = th
        env.humans_state_numpy[i, 5] = 0.0

    env.humans_goals = jnp.array(human_goals)
    env.humans_current_goals = jnp.array(human_goals[:, 0, :])  # start aiming p1->p2
    mujoco.mj_forward(env.model, env.data)

def restart_eval_episode(venv, env, start, waypoints, human_pairs):
    """
    Reset through the VecEnv/VecNormalize, then re-apply:
    - robot start pose + heading
    - first waypoint target
    - endless humans
    Return a fresh *normalized* observation from a 0-action step.
    """
    import numpy as np
    import mujoco

    # Reset the wrapped env (produces normalized obs)
    obs = venv.reset()

    # Robot start pose + heading to first waypoint
    sx, sy = start
    wx, wy = waypoints[0]
    theta0 = float(np.arctan2(wy - sy, wx - sx))
    env.robot_pos = np.array([sx, sy], dtype=np.float32)
    env.robot_theta = theta0
    env.data.qpos[:3] = [sx, sy, theta0]

    # Activate first waypoint, reset baselines
    env.current_wp_index = 0
    env._set_target_position_from_waypoint()
    env.previous_distance = float(np.linalg.norm(env.target_pos - env.robot_pos))
    env._theta_hist.clear()
    for _ in range(3):
        env._theta_hist.append(env.robot_theta)
    mujoco.mj_forward(env.model, env.data)

    # Reinstall endless humans
    set_endless_humans_from_pairs(env, human_pairs)

    # One 0-action step so normalized obs matches teleported state
    zero_action = np.zeros((venv.num_envs, env.action_space.shape[0]), dtype=np.float32)
    obs, _, dones, infos = venv.step(zero_action)

    # Optional: immediate render so viewer catches the new state
    try:
        env.render()
    except Exception:
        pass

    return obs


def main():
    # --- Your data ---
    start = (23.21, 15.81)
    waypoints = [
        (18, 15),
        (8.63, 15.81),   # wp1
        (10, 5.5),
        (9.76, -0.4),
        (10.00, -7.20),   # wp2
        (18.24, -7.1),   # wp3
        (18.24, -15.11),
        (18.24, -23.00),  # wp4
        (8.79, -19.00),  # wp5
        (23.88, -12.16),  # wp6
        (23.88, 15.11),  # wp7
    ]
    human_pairs = [
        ((9.10, 21.80), (9.10,  9.70)),    # humans1
        ((4.18,  4.00), (13.50, 4.00)),    # humans2
        ((13.50, -0.40), (1.70, -0.40)),   # humans3
        ((0.81, -18.25), (0.82, -7.30)),   # humans4
        ((1.38, -13.80), (14.53, -13.80)), # humans5
        ((25.16, -8.60), (25.16, 7.50)),   # humans6
        ((24.81, 9.27),  (24.81, -8.34)),  # humans7
        ((10.3, -21.80), (10.50, -9.70)),   # humans8
        ((26.90, 11.88),  (26.8, -10.00)),     # humans9
        ((22.00, -9.00), (32.00, 7.00)), # humans10
        ((22, 7), (31, -4.6)),
        ((23.08, -15.1),(8.72, -15.19)),
    ]


    def make_env():
        return light_hamrrln(training=False, render_mode="human", n_humans=len(human_pairs))
    # --- Env (eval mode) ---
    # --- make vec env + normalization exactly like training ---
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load("/home/LABAUT/alberto_vaglio/HumanAwareRLNavigation/logs/TENSORBOARD/TQCinversoCURR.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    env = venv.envs[0]
    env.max_episode_time = 1800.0
    env.set_real_time_factor(100)

    # helpful for eval so it doesn’t pirouette in place
    env.disable_eval_safety_stop = True

    # multi-waypoints (persistent across resets)
    env.set_waypoints(waypoints)

    # 1) reset THROUGH the vectorized wrapper (normalized obs)
    obs = venv.reset()

    # 2) NOW apply custom start pose and humans on the underlying env
    sx, sy = start
    wx, wy = waypoints[0]
    theta0 = float(np.arctan2(wy - sy, wx - sx))
    env.robot_pos = np.array([sx, sy], dtype=np.float32)
    env.robot_theta = theta0
    env.data.qpos[:3] = [sx, sy, theta0]
    env._set_target_position_from_waypoint()
    env.previous_distance = float(np.linalg.norm(env.target_pos - env.robot_pos))
    env._theta_hist.clear()
    for _ in range(3):
        env._theta_hist.append(env.robot_theta)
    mujoco.mj_forward(env.model, env.data)

    set_endless_humans_from_pairs(env, human_pairs)

    # 3) force a fresh, normalized observation by doing ONE no-op step
    zero_action = np.zeros((venv.num_envs, env.action_space.shape[0]), dtype=np.float32)
    obs, _, dones, infos = venv.step(zero_action)

    # 4) load model on the SAME venv and run
    model = TQC.load("/home/LABAUT/alberto_vaglio/HumanAwareRLNavigation/TQCinversoCURR", env=venv)

    # Start (or restart) a fresh episode
    # Start (or restart) a fresh episode
    obs = restart_eval_episode(venv, env, start, waypoints, human_pairs)

    attempt = 1
    i = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = venv.step(action)

        # smooth viewing
        if i % 2 == 0:
            env.render()

        info0 = infos[0]
        if "current_waypoint_index" in info0:
            print(f"Reached waypoint #{info0['current_waypoint_index']} / {len(waypoints)}")

        if bool(dones[0]):
            result = info0.get("episode_result", "n/a")
            print(f"Episode ended with: {result}")

            if result in {"collision", "success", "timeout"}:
                attempt += 1
                print(f"→ Restarting from the beginning (attempt {attempt})")
                obs = restart_eval_episode(venv, env, start, waypoints, human_pairs)
                i = 0
                continue

            # (unlikely) any other result → stop
            break

        i += 1




if __name__ == "__main__":
    main()
