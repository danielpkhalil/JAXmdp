import jax
import numpy as np
import matplotlib.pyplot as plt
from gymnax.wrappers.purerl import LogWrapper  # note: removed FlattenObservationWrapper
from gymnax_env import create_tabular_env

# Initialize environment
rng = jax.random.PRNGKey(0)
env = create_tabular_env("consolidated.npz")
env = LogWrapper(env)

# Override the default parameters to use screen observations.
# (TabularEnvParams is a flax.struct.dataclass, so use .replace to update fields.)
env_params = env.default_params.replace(use_screen_observations=True)

# Reset the environment
obs, state = env.reset_env(rng, env_params)
print(f"Initial Observation (array):\n{obs}")
print(f"Initial State: {state}\n")

# If the observation is an image, display it.
if hasattr(obs, "ndim") and obs.ndim == 3:
    plt.figure()
    plt.imshow(np.array(obs))
    plt.title("Initial Observation")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(1.0)
    plt.close()

# List to store rewards
rewards = []

# Step through the environment
for i in range(10):
    rng, key = jax.random.split(rng)
    action = jax.random.randint(key, (), 0, env.num_actions)
    obs, state, reward, done, info = env.step_env(rng, state, action, env_params)
    rewards.append(reward)  # Store the reward
    print(f"Step {i + 1}:")
    print(f"Action taken: {action}")
    print(f"Observation (array):\n{obs}")
    print(f"State: {state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}\n")

    # If observation is a screen image, display it.
    if hasattr(obs, "ndim") and obs.ndim == 3:
        plt.figure()
        plt.imshow(np.array(obs))
        plt.title(f"Observation at Step {i + 1}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(1.0)


# Plotting the rewards over steps
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), np.array(rewards), marker='o')
plt.title('Rewards over Steps')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.grid(True)
plt.show()





# import jax
# import matplotlib.pyplot as plt
# from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
# from gymnax_env import create_tabular_env
#
# # Initialize environment
# rng = jax.random.PRNGKey(0)
# env = create_tabular_env("consolidated.npz")
# env = FlattenObservationWrapper(env)
# env = LogWrapper(env)
#
# # Reset and step through the environment
# env_params = env.default_params
# obs, state = env.reset_env(rng, env_params)
# print(f"Initial Observation: {obs}")
# print(f"Initial State: {state}\n")
#
# # List to store rewards
# rewards = []
#
# for i in range(10):
#     rng, key = jax.random.split(rng)
#     action = jax.random.randint(key, (), 0, env.num_actions)
#     obs, state, reward, done, info = env.step_env(rng, state, action, env_params)
#     rewards.append(reward)  # Store the reward
#     print(f"Step {i + 1}:")
#     print(f"Action taken: {action}")
#     print(f"Observation: {obs}")
#     print(f"State: {state}")
#     print(f"Reward: {reward}")
#     print(f"Done: {done}")
#     print(f"Info: {info}\n")
#
# # Plotting the rewards
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), rewards, marker='o')
# plt.title('Rewards over Steps')
# plt.xlabel('Step')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.show()
#
