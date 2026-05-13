NUM_RAYS = 108 # ERRRRORE minimo 24
N_STACKING = 2
MAX_LIN_VEL_ROBOT = 0.5

MAX_LEN_RAY = 30.0

#DISTANCE_SUCCESS_THRESHOLD = 0.5
MAX_EPISODE_TIME = 100 # seconds, in steps 160/0.25 = 640 steps
ROBOT_RADIUS = 0.25
LIDAR_THRESHOLD = 1.0  # ROBOT_RADIUS * 2
ROBOT_DT = 0.25 # Robot control timestep in seconds
HUMANS_DT = 0.025
N_HUMANS = 20  # Default number of humans in the environment 
PROGRESS_REWARD_SCALE = 0.03  # Scale for progress reward
HUMANS_VELOCITY = 0.0

LIDAR_WEIGHT = 0.1
HUMANS_RADIUS = 0.3