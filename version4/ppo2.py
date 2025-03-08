"""
ppo_tabular_wandb_cnn.py

Usage:
    1) Ensure your local "gymnax_env.py" defines a TabularEnv that returns
       a Box observation shaped like (H, W, C). If it still returns 1D
       vectors, CNN won't make sense. Adjust your environment as necessary.
    2) pip install wandb
    3) python ppo_tabular_wandb_cnn.py

This script:
    - Same PPO code as before
    - Logs results to Weights & Biases
    - Uses a CNN instead of an MLP for the actor-critic
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb   # <-- for logging

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper  # removed FlattenObservationWrapper

# Import your custom environment:
from gymnax_env import TabularEnv, TabularEnvParams

import matplotlib.pyplot as plt


# ------------------------------
# Actor-Critic CNN Module
# ------------------------------
class ActorCritic(nn.Module):
    """
    CNN-based actor-critic for discrete action spaces.
    Assumes input observations are image-like: (H, W, C).
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        """
        x shape: (batch, H, W, C) – or (H, W, C) if batch_size=1.
        Adjust conv layers, kernel sizes, and strides to match your env’s obs shape.
        """
        # Example: two simple conv layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))  # shape: (batch, features)

        # FC layer (example: 64 units)
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        # Policy head (categorical logits)
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        pi = distrax.Categorical(logits=logits)

        # Value head
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(x)
        value = jnp.squeeze(critic, axis=-1)

        return pi, value


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
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # Minibatch size:
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # 1) CREATE ENVIRONMENT (TabularEnv or regular Gymnax)
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    # Remove FlattenObservationWrapper so we keep the shape for CNN
    env = LogWrapper(env)

    # 2) Learning rate schedule
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # 3) TRAIN FUNCTION
    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
        )
        rng, init_rng = jax.random.split(rng)
        # If your tabular env returns shape (H, W, C), confirm it matches:
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(init_rng, init_obs)

        # Define optimizer
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

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # SCAN: Update Steps
        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state

            # 1) Collect a rollout
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                rng, act_rng = jax.random.split(rng)

                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)

                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_rngs, env_state, action, env_params)

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state

            # 2) Compute advantage (GAE)
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1.0 - transition.done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    )
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # 3) PPO Update
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.mean(
                            jnp.maximum(value_losses, value_losses_clipped)
                        )

                        # Policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_normed = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_normed
                        loss_actor2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * gae_normed
                        loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state

                # Shuffle the entire trajectory for minibatches
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)

                # Flatten from [T, N_env, ...] -> [T*N_env, ...]
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    batch,
                )
                # Shuffle
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch,
                )
                # Reshape into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled,
                )

                # Scan over minibatches
                train_state, losses = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metric = traj_batch.info

            # Debug prints if requested
            if config.get("DEBUG"):
                def debug_callback(info):
                    rets = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={rets[t]}")
                jax.debug.callback(debug_callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        # Run the entire training
        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {
            "runner_state": runner_state,
            "metrics": metrics,
        }

    return train


# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    # Define your config
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

        # Activation is removed or replaced in CNN for simplicity,
        # you can add an activation param if you like.

        # Choose "CartPole-v1" or your "TabularMDP"
        "ENV_NAME": "TabularMDP",
        # If using TabularMDP, specify the .npz file
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz",

        # LR schedule
        "ANNEAL_LR": True,

        # Debug
        "DEBUG": False,
    }

    # 1) Initialize wandb logging
    wandb.init(
        project="my_tabular_ppo_cnn",  # project name on wandb
        config=config,                 # store hyperparams in wandb
    )

    rng = jax.random.PRNGKey(42)

    # 2) Create the train function & JIT it
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    # 3) Run training
    out = train_jit(rng)
    print("Training finished. Final output keys:", out.keys())

    # out["metrics"] is a pytree that includes the 'info' logs from the last transitions
    metrics = jax.tree_util.tree_map(lambda x: np.array(x), out["metrics"])

    # The LogWrapper typically logs "returned_episode_returns" with shape [NUM_STEPS, NUM_ENVS, ...]
    if "returned_episode_returns" in metrics:
        try:
            mean_returns = metrics["returned_episode_returns"].mean(axis=-1)
        except:
            mean_returns = metrics["returned_episode_returns"]
        mean_returns = mean_returns.reshape(-1)

        # 4) Log to wandb each step
        for i, ret in enumerate(mean_returns):
            wandb.log({"update_step": i, "mean_return": ret})

        # 5) Plot of these returns
        plt.figure()
        plt.plot(mean_returns, label="Mean Return")
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.title("PPO Training Performance")
        plt.legend()
        plt.tight_layout()

        # Log the figure to wandb
        wandb.log({"training_returns_plot": wandb.Image(plt)})
        plt.show()

    # 6) Finish the wandb run
    wandb.finish()