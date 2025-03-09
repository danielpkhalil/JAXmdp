"""
ppo_tabular_wandb_cnn.py

Usage:
    1) Make sure you have a local "gymnax_env.py" that defines
       TabularEnv and TabularEnvParams (with screen observations).
    2) pip install wandb
    3) python ppo_tabular_wandb_cnn.py

This script:
    - Uses PPO training
    - Automatically picks a CNN if the environment observations are images,
      and an MLP otherwise.
    - Logs results to Weights & Biases
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any, Tuple

import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

# Import your custom environment:
from gymnax_env import TabularEnv, TabularEnvParams

import matplotlib.pyplot as plt


# ------------------------------
# CNN Actor-Critic
# ------------------------------
class CNNActorCritic(nn.Module):
    """
    A small CNN for image observations. It ends with a flatten, then a final MLP layer
    to produce policy logits and value.
    """
    action_dim: int
    activation: str = "relu"

    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

    @nn.compact
    def __call__(self, x):
        """
        x shape: (batch, H, W, C), uint8 in [0..255]
        We'll normalize by 255.0, do a couple of conv layers, then flatten and produce heads.
        """
        # Normalize
        x = x.astype(jnp.float32) / 255.0

        # Convolutional feature extraction
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Optional fully-connected layer
        x = nn.Dense(features=256, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = self.activation_fn(x)

        # Policy head
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                                bias_init=constant(0.0))(x)
        pi = distrax.Categorical(logits=actor_logits)

        # Value head
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return pi, jnp.squeeze(critic, axis=-1)


# ------------------------------
# MLP Actor-Critic
# ------------------------------
class MLPActorCritic(nn.Module):
    """Actor-critic model for 1D (non-image) discrete action spaces."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Flatten if needed (e.g., shape (batch, 1))
        x = x.reshape((x.shape[0], -1))

        # Policy
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

    # Because we only care about TabularEnv with screens, we won't do anything
    # fancy for other Gymnax envs. But if 'ENV_NAME' is CartPole, it might still
    # use the MLP version.

    # Apply wrappers
    # FlattenObservationWrapper is actually optional if you want to keep image shape
    # for TabularEnv with screen observations, so let's comment it out if you prefer
    # env = FlattenObservationWrapper(env)
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
        # Figure out the shape of the observations from the environment
        obs_shape = env.observation_space(env_params).shape
        action_dim = env.action_space(env_params).n

        # Decide if we use CNN or MLP
        # If the obs shape is 3D (e.g. H, W, C), we'll assume they're images
        if len(obs_shape) == 3:
            network = CNNActorCritic(action_dim=action_dim,
                                     activation=config["ACTIVATION"])
        else:
            network = MLPActorCritic(action_dim=action_dim,
                                     activation=config["ACTIVATION"])

        # Initialize network parameters
        rng, init_rng = jax.random.split(rng)
        init_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)  # match env dtype if it's images
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
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,

        # PPO constants
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",  # or "tanh"

        # Choose "CartPole-v1" or your "TabularMDP"
        "ENV_NAME": "TabularMDP",
        # If using TabularMDP with screens, specify the .npz file
        "ENV_FILE": "test.npz",

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
    # By default, LogWrapper keeps track of "returned_episode_returns". We can average them.

    # Convert from device to host numpy
    # metrics = jax.tree_util.tree_map(lambda x: np.array(x), out["metrics"])
    #
    # # The LogWrapper typically logs:
    # #   metrics["returned_episode_returns"] of shape [NUM_STEPS, NUM_ENVS, ...]
    # # or something similar. We'll do a mean across envs per step.
    # if "returned_episode_returns" in metrics:
    #     try:
    #         mean_returns = metrics["returned_episode_returns"].mean(axis=-1)
    #     except Exception:
    #         mean_returns = metrics["returned_episode_returns"]
    #
    #     mean_returns = mean_returns.reshape(-1)  # flatten if it has an extra dimension
    #
    #     # 4) Log to wandb each step
    #     for i, ret in enumerate(mean_returns):
    #         wandb.log({"update_step": i, "mean_return": ret})
    #
    #     # 5) Also log a plot of these returns
    #     plt.figure()
    #     plt.plot(mean_returns, label="Mean Return")
    #     plt.xlabel("Update Step")
    #     plt.ylabel("Return")
    #     plt.title("PPO Training Performance")
    #     plt.legend()
    #     plt.tight_layout()
    #     wandb.log({"training_returns_plot": wandb.Image(plt)})
    #     plt.show()
    #
    # # 6) Finish the wandb run
    # wandb.finish()

    # Faster logging
    # Convert entire metrics pytree to host arrays in one go:
    metrics = jax.device_get(out["metrics"])

    # For instance, if you only need to log every 10th update:
    if "returned_episode_returns" in metrics:
        try:
            mean_returns = metrics["returned_episode_returns"].mean(axis=-1)
        except Exception:
            mean_returns = metrics["returned_episode_returns"]
        mean_returns = mean_returns.reshape(-1)

        # Subsample (e.g. every 10th update)
        subsample = 10
        subsampled_steps = np.arange(0, len(mean_returns), subsample)
        subsampled_returns = mean_returns[::subsample]

        # Log the entire subsampled array in one call
        wandb.log({
            "update_steps": subsampled_steps,
            "mean_returns": subsampled_returns,
        })

        # Plot and log the figure in one go
        plt.figure()
        plt.plot(mean_returns, label="Mean Return")
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.title("PPO Training Performance")
        plt.legend()
        plt.tight_layout()
        wandb.log({"training_returns_plot": wandb.Image(plt)})
        plt.show()

