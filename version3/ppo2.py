#!/usr/bin/env python
import argparse
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
        # If the input is an image (Box observation), cast to float, normalize, and flatten.
        if x.ndim > 2:
            x = x.astype(jnp.float32) / 255.0
            x = x.reshape((x.shape[0], -1))
        # Choose the activation function.
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
# PPO Training and Evaluation Setup
# ------------------------------------------------------------------------------

def make_train(config):
    # Derived configurations.
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    # Create the custom TabularEnv using the provided file path.
    env = create_tabular_env(config["ENV_FILE"])
    env = LogWrapper(env)
    env_params = env.default_params

    # Determine observation shape and action dimension.
    obs_shape = env.observation_space(env_params).shape  # e.g., for Box observations.
    action_dim = env.action_space(env_params).n

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # --- Initialize the network ---
    network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
    rng = jax.random.PRNGKey(config.get("SEED", 30))
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((config["NUM_ENVS"],) + obs_shape, dtype=jnp.uint8)
    network_params = network.init(init_rng, init_x)

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
    rng, env_rng = jax.random.split(rng)
    reset_rng = jax.random.split(env_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_rng, env_params)

    # --------------------------------------------------------------------------
    # Single Training Update (to be jitted and called iteratively)
    # --------------------------------------------------------------------------
    def update_step(runner_state, unused):
        # Unpack runner_state: (train_state, env_state, last_obs, rng)
        train_state, env_state, last_obs, rng = runner_state

        # Collect trajectories over NUM_STEPS.
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            rng, sample_rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, last_obs)
            action = pi.sample(seed=sample_rng)
            log_prob = pi.log_prob(action)

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step_env, in_axes=(0, 0, 0, None)
            )(step_rngs, env_state, action, env_params)
            transition = Transition(done, action, value, reward, log_prob, last_obs, info)
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
        train_state, env_state, last_obs, rng = runner_state
        _, last_val = network.apply(train_state.params, last_obs)

        # Compute advantages using GAE.
        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(carry, transition):
                gae, next_value = carry
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
        def _update_epoch(train_state, batch):
            traj_batch, gae, targets = batch

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

        def _update_epoch_loop(update_state, unused):
            train_state, traj_batch, gae, targets, rng = update_state
            rng, perm_rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            permutation = jax.random.permutation(perm_rng, batch_size)
            batch = (traj_batch, gae, targets)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(_update_epoch, train_state, minibatches)
            update_state = (train_state, traj_batch, gae, targets, rng)
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch_loop, update_state, None, config["UPDATE_EPOCHS"])
        train_state = update_state[0]
        metric = traj_batch.info  # training metrics (can be extended)
        new_runner_state = (train_state, env_state, last_obs, rng)
        return new_runner_state, metric

    # Return the initial runner_state and useful objects.
    initial_runner_state = (train_state, env_state, obsv, rng)
    return initial_runner_state, update_step, env, env_params, network

def evaluate_policy(runner_state, config, env, env_params, network):
    """
    Run a single evaluation episode using the deterministic (mode) action
    at each state. This function is not jitted because it runs once every 1000 updates.
    """
    train_state, _, _, rng = runner_state
    rng, eval_rng = jax.random.split(rng)
    eval_rngs = jax.random.split(eval_rng, config["NUM_ENVS"])
    eval_obs, eval_env_state = jax.vmap(env.reset_env, in_axes=(0, None))(eval_rngs, env_params)
    total_rewards = jnp.zeros((config["NUM_ENVS"],))
    done = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

    def eval_step(carry):
        eval_obs, eval_env_state, total_rewards, done, rng = carry
        rng, step_rng = jax.random.split(rng)
        # Use the deterministic action: choose the mode (i.e. highest probability)
        pi, _ = network.apply(train_state.params, eval_obs)
        action = pi.mode()
        step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
        eval_obs, eval_env_state, reward, done_step, info = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None)
        )(step_rngs, eval_env_state, action, env_params)
        total_rewards = total_rewards + reward * (1 - done.astype(jnp.int32))
        done = jnp.logical_or(done, done_step)
        return (eval_obs, eval_env_state, total_rewards, done, rng)

    state = (eval_obs, eval_env_state, total_rewards, done, rng)
    # Run until every environment signals done.
    while not bool(jnp.all(state[3])):
        state = eval_step(state)
    final_rewards = state[2]
    return float(jnp.mean(final_rewards))

# ------------------------------------------------------------------------------
# Main entry point: Command-line arguments, training loop, and evaluation logging.
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", type=str, default="consolidated.npz",
                        help="Path to the environment file")
    args = parser.parse_args()

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,            # Adjust as needed.
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 1,     # Adjust as needed.
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "ENV_FILE": args.env_file,
        "SEED": 30,
    }

    # Initialize wandb logging.
    wandb.init(project="PPO_Tabular", config=config)

    # Set up training.
    initial_runner_state, update_step, env, env_params, network = make_train(config)
    update_step_jit = jax.jit(update_step)

    num_updates = config["NUM_UPDATES"]
    runner_state = initial_runner_state

    for update in range(num_updates):
        runner_state, metric = update_step_jit(runner_state, None)
        # Log training metrics every 100 updates.
        if update % 100 == 0:
            wandb.log({"train_metric": jax.device_get(metric), "update": update})
        # Every 1000 updates, run an evaluation episode with deterministic actions.
        if update % 1000 == 0:
            eval_reward = evaluate_policy(runner_state, config, env, env_params, network)
            wandb.log({"eval_reward": eval_reward, "update": update})
            print(f"Update {update}: Eval Reward: {eval_reward}")

    print("Training finished.")

if __name__ == "__main__":
    main()
