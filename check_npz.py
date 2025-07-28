import numpy as np  

data = np.load("expert_data/expert_data.npz")

actions = data["actions"]
observations = data["observations"]

with open('all_actions.txt', 'w') as f:
    for action in actions:
        f.write(f"{action}\n")

with open('all_observations.txt', 'w') as f:
    for observation in observations[:1000]:  # Write only the first 10000 observations
        f.write(f"{observation}\n")