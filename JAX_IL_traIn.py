import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Any
import pickle
import os
from functools import partial

# Handle JAX version compatibility
try:
    from jax import tree_util
    tree_leaves = tree_util.tree_leaves
except ImportError:
    # Fallback for older JAX versions
    tree_leaves = jax.tree_leaves

# ============
# Load Expert Data
# ============
data = np.load("expert_data/expert_data.npz")
observations = data["observations"]
actions = data["actions"]

print("Loaded expert data:")
print("Observations:", observations.shape)
print("Actions:", actions.shape)

obs_dim = observations.shape[1]
act_dim = actions.shape[1] if len(actions.shape) > 1 else 1

# Convert to JAX arrays
observations_jax = jnp.array(observations, dtype=jnp.float32)
actions_jax = jnp.array(actions, dtype=jnp.float32)

# ============
# Define Policy Network using Flax
# ============
class PolicyNetwork(nn.Module):
    """Actor-Critic style policy network for behavior cloning."""
    hidden_dims: Tuple[int, ...] = (128, 128)
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, x):
        # Shared feature extraction
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        
        # Policy head (actor)
        policy_logits = nn.Dense(self.action_dim)(x)
        # Apply appropriate activation for action bounds [0,1] for first dim, [-1,1] for second
        action_0 = nn.sigmoid(policy_logits[..., 0:1])  # [0, 1]
        action_1 = nn.tanh(policy_logits[..., 1:2])     # [-1, 1]
        actions = jnp.concatenate([action_0, action_1], axis=-1)
        
        return actions

# ============
# Training State and Functions
# ============
def create_train_state(rng_key, model, obs_dim, learning_rate=3e-4):
    """Create initial training state."""
    dummy_input = jnp.ones((1, obs_dim))
    params = model.init(rng_key, dummy_input)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def mse_loss(params, model, batch_obs, batch_actions):
    """Compute MSE loss between predicted and target actions."""
    pred_actions = model.apply(params, batch_obs)
    loss = jnp.mean((pred_actions - batch_actions) ** 2)
    return loss

def train_step(state, batch_obs, batch_actions, model):
    """Single training step with gradient update."""
    loss_fn = lambda params: mse_loss(params, model, batch_obs, batch_actions)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# JIT compile the train step
train_step_jit = jit(train_step, static_argnames=['model'])

def create_batches(observations, actions, batch_size, rng_key):
    """Create shuffled batches from data."""
    n_samples = observations.shape[0]
    n_batches = n_samples // batch_size
    
    # Shuffle indices
    indices = jax.random.permutation(rng_key, n_samples)
    indices = indices[:n_batches * batch_size]  # Drop remainder
    indices = indices.reshape(n_batches, batch_size)
    
    # Create batches
    batch_obs = observations[indices]
    batch_actions = actions[indices]
    
    return batch_obs, batch_actions

# ============
# Initialize Model and Training State
# ============
rng_key = random.PRNGKey(42)
rng_key, init_key = random.split(rng_key)

model = PolicyNetwork(action_dim=act_dim)
state = create_train_state(init_key, model, obs_dim)

print(f"Model initialized with {sum(x.size for x in tree_leaves(state.params))} parameters")

# ============
# Training Loop
# ============
print("Starting JAX Behavior Cloning training...")

batch_size = 64
epochs = 50

for epoch in range(epochs):
    # Create new batches each epoch with different shuffle
    rng_key, batch_key = random.split(rng_key)
    batch_obs, batch_actions = create_batches(
        observations_jax, actions_jax, batch_size, batch_key
    )
    
    epoch_losses = []
    
    # Process all batches
    for i in range(batch_obs.shape[0]):
        state, loss = train_step_jit(state, batch_obs[i], batch_actions[i], model)
        epoch_losses.append(loss)
    
    avg_loss = jnp.mean(jnp.array(epoch_losses))
    print(f"Epoch {epoch+1:02d}: Loss = {avg_loss:.4f}")

# ============
# Define Inference Function
# ============
def predict_action(params, model, observation):
    """Fast inference function for trained policy."""
    return model.apply(params, observation.reshape(1, -1))[0]

# JIT compile inference
predict_action_jit = jit(predict_action, static_argnames=['model'])

# Create a convenient inference function
def create_policy_fn(state, model):
    """Create a policy function that can be used for deployment."""
    return partial(predict_action_jit, state.params, model)

policy_fn = create_policy_fn(state, model)

# ============
# Save Model
# ============
save_dir = "jax_bc_policy"
os.makedirs(save_dir, exist_ok=True)

# Save the trained parameters and model structure
save_data = {
    'params': state.params,
    'model_config': {
        'hidden_dims': model.hidden_dims,
        'action_dim': model.action_dim,
        'obs_dim': obs_dim
    }
}

with open(os.path.join(save_dir, "policy.pkl"), 'wb') as f:
    pickle.dump(save_data, f)

print(f"✅ Saved JAX Behavior Cloning policy to '{save_dir}/policy.pkl'")

# ============
# Utility Functions for Loading and Testing
# ============
def load_jax_policy(save_path):
    """Load a saved JAX policy."""
    with open(save_path, 'rb') as f:
        save_data = pickle.load(f)
    
    # Recreate model
    config = save_data['model_config']
    model = PolicyNetwork(
        hidden_dims=config['hidden_dims'],
        action_dim=config['action_dim']
    )
    
    # Create inference function
    params = save_data['params']
    policy_fn = partial(predict_action_jit, params, model)
    
    return policy_fn, model, params

# Test the policy on a sample observation
sample_obs = observations_jax[0]
predicted_action = policy_fn(sample_obs)
actual_action = actions_jax[0]

print(f"\nTest prediction:")
print(f"Input observation shape: {sample_obs.shape}")
print(f"Predicted action: {predicted_action}")
print(f"Actual action: {actual_action}")
print(f"Prediction error: {jnp.linalg.norm(predicted_action - actual_action):.4f}")

# ============
# Performance Comparison Demo
# ============
print(f"\n🚀 JAX Performance Benefits:")
print(f"- JIT compilation for faster inference")
print(f"- Vectorized operations across entire dataset")
print(f"- GPU acceleration (if available)")
print(f"- Consistent numerical precision")
print(f"- Easy to integrate with other JAX-based RL libraries")

# Benchmark inference speed
print(f"\nBenchmarking inference speed...")
n_inferences = 1000
sample_batch = observations_jax[:n_inferences]

# Warm up JIT
_ = vmap(policy_fn)(sample_batch)

# Time the inference
import time
start_time = time.time()
predictions = vmap(policy_fn)(sample_batch)
end_time = time.time()

print(f"JAX vectorized inference: {n_inferences} predictions in {end_time - start_time:.4f}s")
print(f"Average per prediction: {(end_time - start_time) / n_inferences * 1000:.2f}ms")