"""
random_env_visualization_matplotlib.py

- Loads an environment (specified by an npz file).
- Resets the environment and steps randomly (no PPO or checkpointing).
- Uses matplotlib to visualize each frame in real time.
- Each frame is shown for a specified pause duration.
- Multiple episodes are run sequentially in the same figure.
- After each episode, the script waits for input to continue.

Usage:
    1) Ensure "gymnax_env.py" is in the same folder.
    2) pip install matplotlib
    3) python random_env_visualization_matplotlib.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
import matplotlib.pyplot as plt

# Import your custom environment (TabularMDP from npz file)
from gymnax_env import TabularEnv, TabularEnvParams

def run_episode(env, env_params, action_dim, rng, max_steps, pause_duration, episode_idx, ax, fig):
    # Reset the environment.
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    # Display the first frame.
    im = ax.imshow(np.array(obs))
    ax.set_title(f"Episode {episode_idx} - Step 0")
    fig.canvas.draw_idle()
    plt.pause(pause_duration)

    # Run the episode.
    for step in range(max_steps):
        rng, action_rng = jax.random.split(rng)
        action = int(jax.random.randint(action_rng, shape=(), minval=0, maxval=action_dim))
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)

        # Update the image.
        im.set_data(np.array(obs))
        ax.set_title(f"Episode {episode_idx} - Step {step+1}")
        fig.canvas.draw_idle()
        plt.pause(pause_duration)

        if done:
            print(f"Episode {episode_idx} finished after {step+1} steps.")
            break
    return rng

def main():
    # Configuration for the environment.
    config = {
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "atlantis_10_fs30.npz",  # update path as needed
        "REWARD_SCALE": 1/21,
        "MAX_STEPS": 10000,         # maximum steps per episode
        "NUM_EPISODES": 5,        # number of episodes to run
        "PAUSE_DURATION": 0.5,    # seconds to pause between frames
    }

    # Create environment using the custom TabularMDP.
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    action_dim = env.action_space(env_params).n
    rng = jax.random.PRNGKey(0)

    # Set up a single interactive matplotlib figure.
    plt.ion()  # Turn on interactive mode.
    fig, ax = plt.subplots()
    plt.show()

    # Run multiple episodes using the same figure.
    for episode in range(1, config["NUM_EPISODES"] + 1):
        print(f"Starting Episode {episode}...")
        rng = run_episode(
            env=env,
            env_params=env_params,
            action_dim=action_dim,
            rng=rng,
            max_steps=config["MAX_STEPS"],
            pause_duration=config["PAUSE_DURATION"],
            episode_idx=episode,
            ax=ax,
            fig=fig
        )
        # Wait for user input to start next episode.
        input(f"Episode {episode} finished. Press Enter to start the next episode (if any)...")
        ax.cla()  # Clear the axes for the next episode.

    plt.ioff()  # Turn off interactive mode.
    plt.show()  # Keep the final frame open.
    print("All episodes completed.")

if __name__ == "__main__":
    main()
