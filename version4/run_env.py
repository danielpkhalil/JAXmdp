import time
import os
import jax
import jax.numpy as jnp
from gymnax_env import TabularEnv, TabularEnvParams, TabularState

# ------------------------------------------------------------------
# Naive rollout: stepping the environment in a Python loop.
# ------------------------------------------------------------------
def naive_rollout(env, params, key, num_steps=1000):
    obs, state = env.reset_env(key, params)
    start_time = time.time()
    steps = 0

    for _ in range(num_steps):
        # Sample a random action using numpy.
        import numpy as np
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
# JIT-friendly versions of the step logic.
# We pass the environment arrays and constants as arguments.
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
    def if_done_fn(_):
        obs = get_obs_jitted(screens, screen_mapping, state.state_idx, params)
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

        reward += jnp.where(
            done_by_horizon & ~done_by_terminal,
            jnp.float32(params.no_done_reward),
            jnp.float32(0.0),
        )
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
def get_obs_jitted(screens, screen_mapping, state_idx: jnp.int32, params: TabularEnvParams):
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
def run_steps_jit(
    transitions,
    rewards,
    screens,
    screen_mapping,
    TERMINAL_STATE,
    initial_state,
    action_seq,
    params
):
    def scan_fn(carry, action):
        state = carry
        obs, next_state, reward, done = step_env_jitted(
            transitions,
            rewards,
            screens,
            screen_mapping,
            TERMINAL_STATE,
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
    # Reset once.
    obs, state = env.reset_env(key, params)
    # Create a batch of random actions in JAX.
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, shape=(num_steps,), minval=0, maxval=env.num_actions)

    start_time = time.time()
    final_state, (obs_seq, rew_seq, done_seq) = run_steps_jit(
        env.transitions,
        env.rewards,
        env.screens,
        env.screen_mapping,
        env.TERMINAL_STATE,
        state,
        actions,
        params
    )
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"[JIT scan] Executed {num_steps} steps in {elapsed:.4f} seconds.")
    return num_steps, elapsed, final_state, obs_seq, rew_seq, done_seq

def main():
    # Specify the NPZ file you already have.
    problem_file = "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz"
    if not os.path.exists(problem_file):
        print(f"Problem file not found: {problem_file}")
        return

    # Instantiate the environment and parameters.
    env = TabularEnv(problem_file)
    params = env.default_params()
    
    # Create a random key.
    key = jax.random.PRNGKey(0)

    # Run the naive (Python loop) rollout.
    naive_steps, naive_time = naive_rollout(env, params, key, num_steps=1000)
    
    # Run the JIT-compiled rollout.
    jit_steps, jit_time, final_state, obs_seq, rew_seq, done_seq = jit_rollout(env, params, key, num_steps=1000)

    print("Final state from JIT run:", final_state)
    if params.use_screen_observations and env.screens is not None:
        print("obs_seq shape (JIT):", obs_seq.shape)
    else:
        print("obs_seq shape (JIT):", obs_seq.shape)

    speedup = naive_time / jit_time if jit_time > 0 else float("inf")
    print(f"Speedup: ~{speedup:.2f}x faster using JIT.")

if __name__ == "__main__":
    main()
