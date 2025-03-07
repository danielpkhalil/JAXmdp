"""
ppo_tabular_wandb_minimal.py

This script:
    - Runs PPO on either your custom TabularEnv or a Gymnax env.
    - Logs training time, plots the final returns, and logs them to wandb.

Ensure you have:
    pip install wandb
    wandb login   # or set WANDB_API_KEY
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any

import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

# Import your custom environment class
from gymnax_env import TabularEnv

import time
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """Actor-critic model for discrete action spaces."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    # Derive training constants
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    # Create environment
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # LR schedule
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # Initialize network
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
        )
        rng, init_rng = jax.random.split(rng)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(init_rng, init_obs)

        # Optimizer
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

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize env
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # Rollout+Update function
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
                    step_rngs, env_state, action, env_params
                )
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                return (train_state, env_state, obsv, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state

            # Compute advantages via GAE
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1.0 - transition.done)
                        - transition.value
                    )
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(
                    _get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # PPO Update
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

                        # Policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_normed = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_normed
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_normed
                        loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)

                # Shuffle entire batch
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    batch,
                )
                shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled,
                )

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


if __name__ == "__main__":
    config = {
        # PPO hyperparams
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e3,
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

        # ENV
        "ENV_NAME": "TabularMDP",  # or "CartPole-v1"
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz",  # if using TabularMDP

        # LR schedule
        "ANNEAL_LR": True,

        # Debug
        "DEBUG": False,
    }

    # -- Initialize wandb
    wandb.init(
        project="my_tabular_ppo_minimal",
        config=config,
    )

    # Build training function & JIT
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    # Run training, measure time
    rng = jax.random.PRNGKey(42)
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    print(f"Training time: {time.time() - t0:.2f} s")

    # Plot final returns
    returns = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    plt.plot(returns)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.show()

    # Log returns to wandb
    for step_i, ret in enumerate(returns):
        wandb.log({"update_step": step_i, "mean_return": float(ret)})

    # Optionally log final figure
    fig = plt.figure()
    plt.plot(returns)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    wandb.log({"Return Plot": wandb.Image(fig)})

    wandb.finish()
