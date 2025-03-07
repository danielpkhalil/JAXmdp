"""
ppo_tabular_wandb.py

Usage:
    1) Make sure you have a local "gymnax_env.py" file that defines
       TabularEnv and TabularEnvParams (with the updated code so it returns
       a Box observation and uses static fields).
    2) pip install wandb
    3) python ppo_tabular_wandb.py

This script:
    - Uses the same PPO code as before.
    - Logs results to Weights & Biases.
    - Includes simple profiling for get_obs and step_env calls.
"""

import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb   # <-- for logging

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any, Union, Optional, Dict, Tuple

import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

# Import your custom environment:
from gymnax_env import TabularEnv, TabularEnvParams

import matplotlib.pyplot as plt

# ------------------------------
# Actor-Critic Module
# ------------------------------
class ActorCritic(nn.Module):
    """Actor-critic model for discrete action spaces."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # Policy (Categorical logits)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Value function
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


# ------------------------------
# Transition NamedTuple
# ------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ------------------------------
# Main Training Function
# ------------------------------
def make_train(config):
    # How many total updates:
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    # Minibatch size:
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    # 1) Create Environment (TabularEnv or regular Gymnax)
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    # Apply wrappers
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # 2) Learning rate schedule
    def linear_schedule(count):
        frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"])
        return config["LR"] * frac

    # 3) Training function
    def train(rng):
        # Initialize network
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
        )
        rng, init_rng = jax.random.split(rng)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(init_rng, init_obs)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # Rollout + PPO update via scan
        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state

            # Rollout
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                rng, act_rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    step_rngs, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                return (train_state, env_state, obsv, rng), transition

            runner_state, traj_batch = jax.lax.scan(_env_step, (train_state, env_state, last_obs, rng), None, config["NUM_STEPS"])
            train_state, env_state, last_obs, rng = runner_state

            # Compute advantages via GAE
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = (transition.reward + config["GAMMA"] * next_value * (1.0 - transition.done) - transition.value)
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(_get_advantages,
                                                   (jnp.zeros_like(last_val), last_val),
                                                   traj_batch,
                                                   reverse=True,
                                                   unroll=16)
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # PPO update
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_normed = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_normed
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_normed
                        loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))
                        entropy = jnp.mean(pi.entropy())
                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled)
                train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            rng = update_state[-1]
            metric = traj_batch.info
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ------------------------------
# Profiling Helpers
# ------------------------------
def profile_get_obs(env: TabularEnv, state, params):
    start = time.time()
    obs = env.get_obs(state, params)
    # Force evaluation
    obs = jax.block_until_ready(obs)
    t = time.time() - start
    print(f"Time for get_obs: {t*1000:.2f} ms")
    return t

def profile_step_env(env: TabularEnv, state, action, params):
    start = time.time()
    # Use env.step directly (bypassing wrappers)
    out = env.step(jax.random.PRNGKey(0), state, action, params)
    # Force evaluation (unpack the tuple)
    out = jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), out)
    t = time.time() - start
    print(f"Time for step_env: {t*1000:.2f} ms")
    return t


# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    config = {
        # PPO hyperparams
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e3,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "TabularMDP",  # or "CartPole-v1"
        # If using TabularMDP, specify the .npz file
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz",
        "ANNEAL_LR": True,
        "DEBUG": False,
    }

    # Initialize wandb logging
    wandb.init(project="my_tabular_ppo", config=config)

    # Create environment instance for profiling
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # Create a dummy state for profiling get_obs and step_env
    from gymnax_env import TabularState  # import state definition
    dummy_state = TabularState(
        state_idx=jnp.array(0, dtype=jnp.int32),
        steps=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
        time=0
    )
    # Profile get_obs
    profile_get_obs(env, dummy_state, env_params)
    # Profile step_env with dummy action (e.g., 0)
    profile_step_env(env, dummy_state, 0, env_params)

    # Create and run training
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    rng = jax.random.PRNGKey(42)
    out = train_jit(rng)
    print("Training finished. Final output keys:", out.keys())

    # Convert metrics from device to host numpy
    metrics = jax.tree_util.tree_map(lambda x: np.array(x), out["metrics"])
    if "returned_episode_returns" in metrics:
        try:
            mean_returns = metrics["returned_episode_returns"].mean(axis=-1)
        except:
            mean_returns = metrics["returned_episode_returns"]
        mean_returns = mean_returns.reshape(-1)
        for i, ret in enumerate(mean_returns):
            wandb.log({"update_step": i, "mean_return": float(ret)})
        plt.figure()
        plt.plot(mean_returns, label="Mean Return")
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.title("PPO Training Performance")
        plt.legend()
        plt.tight_layout()
        wandb.log({"training_returns_plot": wandb.Image(plt)})
        plt.show()

    wandb.finish()