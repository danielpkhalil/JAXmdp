import time
import numpy as np
import jax
from gymnax_env import TabularEnv

# Create a dummy problem file if one doesn't exist.
def create_dummy_problem_file(filename="/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz"):
    num_states = 10
    num_actions = 4
    # Create random transitions between states (values between 0 and num_states-1)
    transitions = np.random.randint(0, num_states, size=(num_states, num_actions))
    # Create random rewards
    rewards = np.random.randn(num_states, num_actions).astype(np.float32)
    # Create dummy screen observations (e.g., 64x64 RGB images)
    screens = np.random.randint(0, 256, size=(num_states, 64, 64, 3), dtype=np.uint8)
    # Mapping from state index to screen index (identity mapping in this dummy example)
    screen_mapping = np.arange(num_states)
    np.savez(filename, transitions=transitions, rewards=rewards, screens=screens, screen_mapping=screen_mapping)
    print(f"Dummy problem file saved as {filename}")

if __name__ == "__main__":
    # Create the dummy problem file.
    problem_file = "dummy_problem.npz"
    create_dummy_problem_file(problem_file)

    # Instantiate the environment with the dummy problem file.
    env = TabularEnv(problem_file)
    params = env.default_params()

    # Create a JAX random key.
    key = jax.random.PRNGKey(0)

    # Reset environment.
    obs, state = env.reset_env(key, params)

    steps = 0
    start_time = time.time()

    # Iterate for 1000 steps.
    for i in range(1000):
        # Sample a random action.
        action = np.random.randint(0, env.num_actions)
        key, subkey = jax.random.split(key)
        # Step the environment.
        obs, state, reward, done, info = env.step_env(subkey, state, action, params)
        steps += 1
        # If the episode ends, reset the environment.
        if done:
            obs, state = env.reset_env(key, params)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Executed {steps} steps in {elapsed:.4f} seconds.")