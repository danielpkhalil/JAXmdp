"""
visualize_policy.py

- Loads the checkpoint saved from training.
- Runs one deterministic evaluation episode with the loaded policy.
- Records and saves each observation frame to create a GIF of the policy rollout.

Usage:
    1) Ensure "gymnax_env.py" is in the same folder.
    2) pip install imageio
    3) python visualize_policy.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
import flax.linen as nn
import distrax
from flax.training import checkpoints
import imageio

# Import your custom environment
from gymnax_env import TabularEnv, TabularEnvParams

# ------------------------------
# Actor-Critic Networks (must match training script)
# ------------------------------
class CNNActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    def setup(self):
        self.activation_fn = nn.relu if self.activation == "relu" else nn.tanh

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2),
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                    bias_init=nn.initializers.constant(0.0))(x)
        x = self.activation_fn(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2),
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                    bias_init=nn.initializers.constant(0.0))(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)
        x = self.activation_fn(x)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)

class MLPActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = act_fn(x)
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = act_fn(x)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)

def main():
    # Configuration matching your training settings.
    config = {
        "SEED": 42,
        "ACTIVATION": "relu",
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "atlantis_10_fs30.npz",
        "REWARD_SCALE": 1/100,
    }

    # 1) Create the environment (here using your custom TabularEnv)
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    # 2) Determine observation shape and action dimension
    obs_shape = env.observation_space(env_params).shape
    action_dim = env.action_space(env_params).n

    # 3) Build the network based on observation shape
    if len(obs_shape) == 3:
        network = CNNActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        network = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    # Initialize network parameters (they will be overwritten by the checkpoint)
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_obs)

    # 4) Load the checkpoint (using an absolute path)
    checkpoint_dir = os.path.abspath("./checkpoints")
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target={"params": params})
    print("Checkpoint loaded.")

    # Unwrap extra "params" layer if present.
    loaded_params = restored_state["params"]
    if "params" in loaded_params:
        loaded_params = loaded_params["params"]

    # 5) Run one deterministic evaluation episode, recording frames
    frames = []
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    done = False
    max_steps = 100000
    steps = 0
    total_reward = 0

    while (not done) and (steps < max_steps):
        frame = np.array(obs)
        frames.append(frame)

        # Use the policy deterministically: choose the action with highest logit
        pi, _ = network.apply({"params": loaded_params}, obs[None, ...])
        action = int(jnp.argmax(pi.logits[0]))
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
        total_reward += reward
        steps += 1

    print(f"Policy rollout completed in {steps} steps.")
    print(f"Total reward: {total_reward}")

    # 6) Save the recorded frames as a GIF with slower playback (1 frame per second)
    imageio.mimsave('policy_rollout.gif', frames)
    # Alternatively, use duration:
    # imageio.mimsave('policy_rollout.gif', frames, duration=1.0)
    print("Saved policy rollout GIF as policy_rollout.gif")

if __name__ == "__main__":
    main()
