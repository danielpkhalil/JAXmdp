"""
compare_ppo_vs_env.py

Measures wall-clock time of:
1) Pure JAX environment stepping (with no policy update)
2) Full PPO training

This helps us see how much overhead the PPO updates add on top
of just stepping through the environment.

Usage:
    python compare_ppo_vs_env.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import optax

from flax.training.train_state import TrainState
from typing import NamedTuple
from functools import partial

import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

import matplotlib.pyplot as plt

# If you have your ActorCritic model in a separate module, import it.
# For a self-contained example, let's just define a basic one here:
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant

class ActorCritic(nn.Module):
    """Minimal example of an Actor-Critic net for discrete actions."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Policy logits
        actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
        actor_hidden = activation(actor_hidden)
        actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(actor_hidden)
        actor_hidden = activation(actor_hidden)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_hidden)
        pi = distrax.Categorical(logits=logits)

        # Value
        critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
        critic_hidden = activation(critic_hidden)
        critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(critic_hidden)
        critic_hidden = activation(critic_hidden)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_hidden)
        value = jnp.squeeze(value, axis=-1)
        return pi, value


# This matches the PPO config, so that both runs do the same total env steps
CONFIG = {
    # PPO hyperparams
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 20_000,   # example scale
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,

    # PPO constants
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",

    # Env name
    "ENV_NAME": "CartPole-v1",

    # LR schedule
    "ANNEAL_LR": True,
}

# ---------------------------------------------------
# We define one chunk of code to do environment steps
# ---------------------------------------------------

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def create_env(config):
    """
    Create the env (wrapped with FlattenObservationWrapper, LogWrapper).
    """
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    return env, env_params


def sample_action_and_step(rng, params, apply_fn, obs, env, env_state, env_params):
    """
    For the current observation, sample an action from the policy,
    then step the env. Return next states and a Transition.
    """
    pi, value = apply_fn(params, obs)
    rng, act_rng = jax.random.split(rng)
    action = pi.sample(seed=act_rng)
    log_prob = pi.log_prob(action)

    rng, step_rng = jax.random.split(rng)
    step_rngs = jax.random.split(step_rng, obs.shape[0])  # for each of NUM_ENVS
    new_obs, new_env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(step_rngs, env_state, action, env_params)

    transition = Transition(
        done, action, value, reward, log_prob, obs, info
    )
    return rng, new_obs, new_env_state, transition


def build_env_scan_fn(num_steps):
    """
    Returns a function that, when scanned over, will step the environment
    for `num_steps`, collecting transitions.
    """

    def env_rollout(runner_state, _):
        train_state, env_state, last_obs, rng, env, env_params = runner_state
        rng, new_obs, new_env_state, transition = sample_action_and_step(
            rng, train_state.params, train_state.apply_fn, last_obs, env, env_state, env_params
        )
        new_runner_state = (train_state, new_env_state, new_obs, rng, env, env_params)
        return new_runner_state, transition

    def run_env_steps(runner_state, _):
        # Inside each "update" we do num_steps of environment stepping
        runner_state, traj_batch = jax.lax.scan(
            env_rollout, runner_state, None, length=num_steps
        )
        return runner_state, traj_batch

    return run_env_steps


# -------------------------------------------------------------------
# 1) Plain Environment Simulation
#    We'll do the same scanning structure, but skip the PPO update
# -------------------------------------------------------------------

def make_env_sim(config):
    """
    Return a function that does only environment stepping (no PPO),
    scanning the same number of times as the PPO code would do.
    """

    # 1) Create env
    env, env_params = create_env(config)

    # 2) Compute how many times to do the rollouts
    num_updates = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # 3) Minimal "policy" just to keep code consistent:
    #    We'll define a dummy network with random params.
    rng = jax.random.PRNGKey(0)
    dummy_net = ActorCritic(env.action_space(env_params).n, config["ACTIVATION"])
    init_obs = jnp.zeros(env.observation_space(env_params).shape)
    params = dummy_net.init(rng, init_obs)
    train_state = TrainState.create(
        apply_fn=dummy_net.apply,
        params=params,
        tx=optax.sgd(0.0),  # no learning
    )

    # 4) Reset the environment across config["NUM_ENVS"]
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    # 5) Define a scan that does the environment steps (no PPO).
    rollout_scan = build_env_scan_fn(config["NUM_STEPS"])

    def run_env_only(rng):
        runner_state = (train_state, env_state, obsv, rng, env, env_params)

        def do_update(runner_state, _):
            # Just collect environment steps
            runner_state, traj_batch = rollout_scan(runner_state, None)
            return runner_state, traj_batch

        runner_state, all_trajs = jax.lax.scan(
            do_update, runner_state, None, length=num_updates
        )
        return all_trajs  # ignoring final runner_state

    return jax.jit(run_env_only)


# -------------------------------------------------------------------
# 2) Full PPO Implementation
#    The same structure from your original code
# -------------------------------------------------------------------

def linear_schedule(count, config):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / (
            config["TOTAL_TIMESTEPS"]
            // config["NUM_STEPS"]
            // config["NUM_ENVS"]
        )
    )
    return config["LR"] * frac


def make_ppo_train(config):
    """
    Return a function that does the full PPO training.
    (Essentially your existing PPO logic, but minimized a bit.)
    """
    # Setup env
    env, env_params = create_env(config)
    action_dim = env.action_space(env_params).n

    # Setup network & optimizer
    net = ActorCritic(action_dim, config["ACTIVATION"])
    rng = jax.random.PRNGKey(123)
    init_obs = jnp.zeros(env.observation_space(env_params).shape)
    params = net.init(rng, init_obs)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lambda c: linear_schedule(c, config), eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    train_state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
    )

    # Reset env
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    # Count how many updates
    num_updates = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # We'll replicate the key PPO steps: rollout + GAE + update
    # For brevity, define them as sub-functions here.

    def rollout(runner_state, _):
        """
        Collect a batch of environment transitions for PPO.
        """
        train_state, env_state, obs, rng = runner_state

        def _env_step(runner_state, _):
            train_state, env_state, obs, rng = runner_state
            rng, new_obs, new_env_state, transition = sample_action_and_step(
                rng,
                train_state.params,
                train_state.apply_fn,
                obs,
                env,
                env_state,
                env_params
            )
            return (train_state, new_env_state, new_obs, rng), transition

        (train_state, env_state, obs, rng), traj_batch = jax.lax.scan(
            _env_step, (train_state, env_state, obs, rng), None, config["NUM_STEPS"]
        )
        return (train_state, env_state, obs, rng), traj_batch

    def gae_advantages(traj_batch, last_val):
        """
        Compute GAE (generalized advantage estimation).
        """
        def _get_adv(carry, t):
            gae, next_val = carry
            delta = (
                t.reward
                + config["GAMMA"] * next_val * (1.0 - t.done)
                - t.value
            )
            gae = (
                delta
                + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - t.done) * gae
            )
            return (gae, t.value), gae

        (_, _), advantages = jax.lax.scan(
            _get_adv,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        returns = advantages + traj_batch.value
        return advantages, returns

    def update_ppo(train_state, traj_batch, advantages, targets, rng):
        """
        Do the PPO update epochs on the given trajectory/advantages.
        """
        batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]

        # Flatten trajectory (T, N_env) -> (T*N_env)
        def flatten_fn(x):
            return x.reshape((batch_size,) + x.shape[2:])

        flat_traj = jax.tree_util.tree_map(flatten_fn, traj_batch)
        flat_adv = advantages.reshape((batch_size,))
        flat_tgt = targets.reshape((batch_size,))

        def loss_fn(params, mb_traj, mb_adv, mb_tgt):
            pi, value = train_state.apply_fn(params, mb_traj.obs)
            logp = pi.log_prob(mb_traj.action)
            ratio = jnp.exp(logp - mb_traj.log_prob)

            # Value loss (clip)
            value_clipped = mb_traj.value + (
                value - mb_traj.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            v_loss_1 = (value - mb_tgt) ** 2
            v_loss_2 = (value_clipped - mb_tgt) ** 2
            value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

            # Policy loss (clip)
            adv_normed = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            loss_actor1 = ratio * adv_normed
            loss_actor2 = jnp.clip(
                ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
            ) * adv_normed
            loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

            # Entropy
            entropy = jnp.mean(pi.entropy())

            total_loss = (
                loss_actor
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF"] * entropy
            )
            return total_loss

        def do_epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)

            # Shuffle
            shuf_traj = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                flat_traj
            )
            shuf_adv = jnp.take(flat_adv, permutation, axis=0)
            shuf_tgt = jnp.take(flat_tgt, permutation, axis=0)

            # Minibatches
            mb_size = batch_size // config["NUM_MINIBATCHES"]
            def minibatch_slice(x, i):
                return x[i*mb_size:(i+1)*mb_size]

            def update_minibatch(train_state, i):
                mb_tr = jax.tree_util.tree_map(
                    lambda x: minibatch_slice(x, i), shuf_traj
                )
                mb_adv = minibatch_slice(shuf_adv, i)
                mb_tgt = minibatch_slice(shuf_tgt, i)

                grads = jax.grad(loss_fn)(train_state.params, mb_tr, mb_adv, mb_tgt)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, None

            # Scan over minibatches
            train_state, _ = jax.lax.scan(update_minibatch, train_state, jnp.arange(config["NUM_MINIBATCHES"]))
            return (train_state, rng), None

        # Scan over PPO epochs
        init_carry = (train_state, rng)
        (train_state, rng), _ = jax.lax.scan(do_epoch, init_carry, None, config["UPDATE_EPOCHS"])
        return train_state, rng

    @partial(jax.jit, static_argnums=0)
    def train_loop(rng):
        train_st = train_state
        st_env = env_state
        obs_ = obsv

        def ppo_update(runner_state, update_idx):
            train_st, st_env, obs_, rng = runner_state

            # 1) Rollout
            (train_st2, st_env2, obs2, rng2), traj_batch = rollout((train_st, st_env, obs_, rng), None)

            # 2) Last value for GAE
            _, last_val = train_st2.apply_fn(train_st2.params, obs2)
            advantages, targets = gae_advantages(traj_batch, last_val)

            # 3) Update
            train_st3, rng3 = update_ppo(train_st2, traj_batch, advantages, targets, rng2)

            return (train_st3, st_env2, obs2, rng3), None

        num_updates = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        init_state = (train_st, st_env, obs_, rng)
        (train_st_final, st_env_final, obs_final, rng_final), _ = jax.lax.scan(
            ppo_update,
            init_state,
            jnp.arange(num_updates),
        )
        return (train_st_final, st_env_final, obs_final, rng_final)

    return train_loop


# -------------------------------------------------------------------
# Main to compare timings
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Plain env stepping
    env_run_fn = make_env_sim(CONFIG)
    rng = jax.random.PRNGKey(999)
    t0 = time.perf_counter()
    _ = env_run_fn(rng)  # run all environment steps
    t1 = time.perf_counter()
    env_sim_time = t1 - t0

    # 2) Full PPO training
    ppo_fn = make_ppo_train(CONFIG)
    rng = jax.random.PRNGKey(42)
    t2 = time.perf_counter()
    _ = ppo_fn(rng)      # run all environment steps + PPO updates
    t3 = time.perf_counter()
    ppo_time = t3 - t2

    print("\n=====================")
    print("Plain Env Sim Time   :", env_sim_time, "seconds")
    print("Full PPO Training Time:", ppo_time, "seconds")
    print("=====================")

    # Optionally you can also compare step rates, etc.
    total_steps = CONFIG["TOTAL_TIMESTEPS"]
    print(f"Plain env throughput : {int(total_steps/env_sim_time)} steps/sec (approx).")
    print(f"PPO throughput       : {int(total_steps/ppo_time)} steps/sec (approx).")
