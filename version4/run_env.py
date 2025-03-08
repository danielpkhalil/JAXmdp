import time
import numpy as np
import jax
import jax.numpy as jnp

from gymnax_env import TabularEnv, TabularEnvParams, TabularState

def create_dummy_problem_file(filename="dummy_problem.npz"):
    """
    Create a random MDP .npz file with transitions, rewards, and screen data.
    """
    num_states = 10
    num_actions = 4
    transitions = np.random.randint(0, num_states, size=(num_states, num_actions))
    rewards = np.random.randn(num_states, num_actions).astype(np.float32)
    # Suppose each state has a 64x64x3 "screen" observation
    screens = np.random.randint(
        0, 256, size=(num_states, 64, 64, 3), dtype=np.uint8
    )
    screen_mapping = np.arange(num_states)
    np.savez(
        filename,
        transitions=transitions,
        rewards=rewards,
        screens=screens,
        screen_mapping=screen_mapping
    )
    print(f"Dummy problem file saved as {filename}")

def naive_rollout(env, params, key, num_steps=1000):
    """
    Naive Python loop that steps the environment each time.
    """
    obs, state = env.reset_env(key, params)
    start_time = time.time()
    steps = 0

    for _ in range(num_steps):
        # Random action from Python/numpy
        action = np.random.randint(0, env.num_actions)
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = env.step_env(subkey, state, action, params)
        steps += 1
        if done:
            obs, state = env.reset_env(key, params)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"[Naive loop] Executed {steps} steps in {elapsed:.4f} seconds.")
    return steps, elapsed

# ------------------------------------------------------------------
# Below we define a JIT-friendly version of step_env
# that references environment arrays in a single function.
# We'll use a 'scan' to run multiple steps in one device call.
# ------------------------------------------------------------------
@jax.jit
def step_env_jitted(
    transitions,
    rewards,
    screens,
    screen_mapping,
    TERMINAL_STATE,
    state: TabularState,
    action: jnp.int32,
    params: TabularEnvParams,
):
    """
    A JIT-compatible version of step logic that references environment arrays.
    """
    def if_done_fn(_):
        obs = get_obs_jitted(
            screens, screen_mapping, state.state_idx, params
        )
        reward = jnp.float32(0.0)
        return obs, state, reward, state.done

    def if_not_done_fn(_):
        next_state_idx = transitions[state.state_idx, action]
        reward = rewards[state.state_idx, action]

        new_steps = state.steps + 1
        done_by_terminal = (next_state_idx == TERMINAL_STATE)
        done_by_horizon = (new_steps >= params.horizon)
        done_by_reward = (reward != 0) & (params.done_on_reward)
        done_new = done_by_terminal | done_by_horizon | done_by_reward

        # If horizon ended (and not terminal), add no_done_reward
        reward += jnp.where(
            done_by_horizon & ~done_by_terminal,
            jnp.float32(params.no_done_reward),
            jnp.float32(0.0),
        )

        # Freeze the state_idx if done
        next_state_idx = jnp.where(done_new, state.state_idx, next_state_idx)
        next_state = TabularState(
            state_idx=next_state_idx,
            steps=new_steps,
            done=done_new,
            time=state.time + 1
        )
        obs = get_obs_jitted(screens, screen_mapping, next_state_idx, params)
        return obs, next_state, reward, done_new

    obs, next_state, reward, done_new = jax.lax.cond(
        state.done, if_done_fn, if_not_done_fn, operand=None
    )
    return obs, next_state, reward, done_new

@jax.jit
def get_obs_jitted(
    screens,
    screen_mapping,
    state_idx: jnp.int32,
    params: TabularEnvParams,
):
    """
    Return a screen or a [state_idx].
    """
    # If we want screen obs
    if params.use_screen_observations and (screens is not None):
        def valid_screen_fn(idx):
            return screens[screen_mapping[idx]]
        def invalid_screen_fn(_):
            return jnp.zeros(screens.shape[1:], dtype=jnp.uint8)
        return jax.lax.cond(
            (state_idx >= 0) & (state_idx < screens.shape[0]),
            valid_screen_fn,
            invalid_screen_fn,
            state_idx
        )
    else:
        return jnp.array([state_idx], dtype=jnp.float32)

@jax.jit
def run_steps_jit(env, initial_state, action_seq, params):
    """
    Run the environment for len(action_seq) steps using a single jax.lax.scan.
    Returns (final_state, (obs_seq, reward_seq, done_seq))
    """

    def scan_fn(carry, action):
        state = carry
        obs, next_state, reward, done = step_env_jitted(
            env.transitions,
            env.rewards,
            env.screens,
            env.screen_mapping,
            env.TERMINAL_STATE,
            state,
            action,
            params
        )
        return next_state, (obs, reward, done)

    final_state, (obs_seq, rew_seq, done_seq) = jax.lax.scan(
        scan_fn, initial_state, action_seq
    )
    return final_state, (obs_seq, rew_seq, done_seq)

def jit_rollout(env, params, key, num_steps=1000):
    """
    A JIT-compiled rollout that performs all steps in a single device call.
    """
    # Reset once
    obs, state = env.reset_env(key, params)

    # Create a batch of random actions in JAX
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, shape=(num_steps,), minval=0, maxval=env.num_actions)

    # Run a single jitted call
    start_time = time.time()
    final_state, (obs_seq, rew_seq, done_seq) = run_steps_jit(env, state, actions, params)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"[JIT scan] Executed {num_steps} steps in {elapsed:.4f} seconds.")
    return num_steps, elapsed, final_state, obs_seq, rew_seq, done_seq

def main():
    # 1) Create dummy data
    problem_file = "dummy_problem.npz"
    create_dummy_problem_file(problem_file)

    # 2) Instantiate environment
    env = TabularEnv(problem_file)
    params = env.default_params()

    # 3) Create a random key
    key = jax.random.PRNGKey(0)

    # 4) Naive Python stepping
    naive_steps, naive_time = naive_rollout(env, params, key, num_steps=1000)

    # 5) JIT-compiled stepping
    jit_steps, jit_time, final_state, obs_seq, rew_seq, done_seq = jit_rollout(env, params, key, num_steps=1000)

    print("Final state from JIT run:", final_state)
    if params.use_screen_observations and env.screens is not None:
        # obs_seq shape ~ (num_steps, 64, 64, 3)
        print("obs_seq shape (JIT):", obs_seq.shape)
    else:
        # obs_seq shape ~ (num_steps, 1)
        print("obs_seq shape (JIT):", obs_seq.shape)

    # 6) Print speed comparison
    speedup = naive_time / jit_time if jit_time > 0 else float("inf")
    print(f"Speedup: ~{speedup:.2f}x faster using JIT.")

if __name__ == "__main__":
    main()