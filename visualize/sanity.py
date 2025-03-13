"""
sanity_check_env.py

This script loads your TabularMDP environment from the npz file and performs several sanity checks:
 - Prints out key properties read from the npz file (e.g., shapes of transitions, rewards, screens).
 - Prints the default environment parameters.
 - Shows the action and observation spaces.
 - Runs one episode with random actions.
 - At each step, prints detailed info (current state, action, reward, done flags, info dictionary).
 - Visualizes the observation via matplotlib in real time.

Usage:
    1) Ensure "gymnax_env.py" is in the same folder.
    2) pip install matplotlib
    3) python sanity_check_env.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
import matplotlib.pyplot as plt

# Import your custom environment (TabularMDP from npz file)
from gymnax_env import TabularEnv, TabularEnvParams


def sanity_check():
    # Configuration for the environment.
    config = {
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "atlantis_20_fs30.npz",  # update path as needed
        "REWARD_SCALE": 1,
        "MAX_STEPS": 10000,  # maximum steps per episode
        "PAUSE_DURATION": 1.0,  # seconds to pause between frames for visualization
    }

    # Create environment using the custom TabularMDP.
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
    else:
        env, _ = gymnax.make(config["ENV_NAME"])

    # Print out information read from the npz file.
    print("Transitions shape:", env.transitions.shape)
    print("Rewards shape:", env.rewards.shape)
    if env.screens is not None:
        print("Screens shape:", env.screens.shape)
    if env.screen_mapping is not None:
        print("Screen mapping shape:", env.screen_mapping.shape)

    # Get default parameters and update reward scale.
    env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    print("\nDefault environment parameters:")
    print(env_params)

    # Print the action and observation spaces.
    action_space = env.action_space(env_params)
    observation_space = env.observation_space(env_params)
    print("\nAction space:", action_space)
    print("Observation space:", observation_space)

    # Initialize a random key.
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)

    # Reset the environment.
    obs, state = env.reset_env(reset_rng, env_params)
    print("\nInitial state:", state)
    print("Initial observation:", obs)

    # Set up matplotlib for interactive display.
    plt.ion()  # Turn on interactive mode.
    fig, ax = plt.subplots()
    im = ax.imshow(np.array(obs))
    ax.set_title("Step 0")
    plt.show()
    actions = [1,3,2,2,1,5,2,2,1,2,2] + [0]*1000
    # Run an episode with random actions.
    for step in range(config["MAX_STEPS"]):
        rng, action_rng = jax.random.split(rng)
        # Sample a random action from the action space.
        action = int(jax.random.randint(action_rng, shape=(), minval=0, maxval=action_space.n))
        #new action
        action = actions[step]
        print(f"\nStep {step + 1}:")
        print("  Action taken:", action)

        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step_env(step_rng, state, action, env_params)

        # Print detailed information.
        print("  New state:", state)
        print("  Reward:", reward)
        print("  Done flag:", done)
        print("  Info:", info)

        # Update the visualization.
        im.set_data(np.array(obs))
        ax.set_title(f"Step {step + 1} | Action: {action} | Reward: {reward:.2f} | Done: {done}")
        fig.canvas.draw_idle()
        plt.pause(config["PAUSE_DURATION"])

        if done:
            print("\nEpisode finished.")
            break

    plt.ioff()  # Turn off interactive mode.
    plt.show()  # Keep the final frame open.


if __name__ == "__main__":
    sanity_check()
