#!/usr/bin/env python
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import wandb

# Import your custom TabularEnv creation function and wrappers
from gymnax_env import create_tabular_env
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

# ------------------------------------------------------------------------------
# Actor-Critic Network Definition
# ------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        # If the input is an image (Box observation), cast to float,
        # normalize to [0, 1], and flatten.
        if x.ndim > 2:
            x = x.astype(jnp.float32) / 255.0
            x = x.reshape((x.shape[0], -1))
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        # --- Actor network ---
        actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_hidden = activation_fn(actor_hidden)
        actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_hidden)
        actor_hidden = activation_fn(actor_hidden)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_hidden)
        pi = distrax.Categorical(logits=logits)

        # --- Critic network ---
        critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic_hidden = activation_fn(critic_hidden)
        critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic_hidden)
        critic_hidden = activation_fn(critic_hidden)
        critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_hidden)
        critic_value = jnp.squeeze(critic_value, axis=-1)

        return pi, critic_value

# ------------------------------------------------------------------------------
# Transition NamedTuple for storing trajectories
# ------------------------------------------------------------------------------

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any

# ------------------------------------------------------------------------------
# PPO Training Function
# ------------------------------------------------------------------------------

def make_train(config):
    # Derived configurations
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"]))
    config["MINIBATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    # Create the custom TabularEnv
    env = create_tabular_env("consolidated.npz")
    # If using screen observations, do NOT flatten them here so the network can process images.
    # env = FlattenObservationWrapper(env)  # Not used for image observations.
    env = LogWrapper(env)
    env_params = env.default_params

    # Determine observation shape and action dimension.
    obs_shape = env.observation_space(env_params).shape  # For Box observations.
    action_dim = env.action_space(env_params).n

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # --- Initialize the network ---
        network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((config["NUM_ENVS"],) + obs_shape, dtype=jnp.uint8)
        network_params = network.init(_rng, init_x)

        # --- Set up the optimizer ---
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

        # --- Initialize the environment ---
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_rng, env_params)

        # --- Training loop ---
        def _update_step(runner_state, unused):
            # Collect trajectories over NUM_STEPS time steps.
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step_env, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            # Compute advantages using Generalized Advantage Estimation (GAE).
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Update the network with collected trajectories.
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, gae, targets = batch_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux_vals), grads = grad_fn(train_state.params, traj_batch, gae, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, gae, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], "batch size must equal num_steps * num_envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, gae, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, gae, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]

            # --- Minimal modification: compute cumulative reward over NUM_STEPS ---
            reward_sum = jnp.sum(traj_batch.reward)
            metric = reward_sum  # You can include more info if needed.

            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,  # 4
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 1,  # 4
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
    }

    # Initialize wandb (logging happens after training here)
    wandb.init(project="PPO_WandB_Logger", config=config)

    rng = jax.random.PRNGKey(30)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    out = train_jit(rng)
    print("Training finished. Metrics:")
    print(out["metrics"])

    # After training, use the returned metrics (an array with one entry per update)
    metrics = out["metrics"]
    for update, reward in enumerate(metrics):
        current_timestep = update * config["NUM_ENVS"] * config["NUM_STEPS"]
        wandb.log({"timestep": int(current_timestep), "reward": float(reward)})
