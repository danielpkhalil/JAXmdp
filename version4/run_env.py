"""
ppo_vs_env_benchmark.py

This script benchmarks two things on the CartPole-v1 environment:
1. Running PPO training (using your PPO code that logs to wandb)
2. Just iterating through the environment with random actions

It also logs a reward plot from the PPO training (using LogWrapper) to wandb.
Extra print/debug statements have been added so you can see what is taking time.
Usage:
    pip install jax jaxlib gymnax flax optax wandb distrax matplotlib
    python ppo_vs_env_benchmark.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
import matplotlib.pyplot as plt
import optax
import wandb
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

# Import your custom environment if needed (for TabularMDP)
from gymnax_env import TabularEnv, TabularEnvParams  # adjust or remove if not using TabularMDP

# ------------------------------
# Actor-Critic Module (same as your PPO code)
# ------------------------------
class ActorCritic(nn.Module):
    """Actor-critic model for discrete action spaces."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # Policy network (compute categorical logits)
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
from typing import NamedTuple
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ------------------------------
# PPO Training Function (same as your PPO code, with extra prints)
# ------------------------------
def make_train(config):
    # Compute number of update steps and minibatch size
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    # Create environment
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    # Apply wrappers
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # Linear learning rate schedule
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # DEBUG: Starting training function
        jax.debug.print(">> Starting training function at time: {}", time.time())

        # Initialize network
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
        )
        rng, init_rng = jax.random.split(rng)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(init_rng, init_obs)
        jax.debug.print(">> Network initialized at time: {}", time.time())

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
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        jax.debug.print(">> Optimizer and train state created at time: {}", time.time())

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
        jax.debug.print(">> Environment reset complete at time: {}", time.time())

        # Main update loop
        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            jax.debug.print(">> Starting an update step at time: {}", time.time())

            # Rollout collection
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
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state
            jax.debug.print(">> Rollout collection complete at time: {}", time.time())

            # Compute advantage (GAE)
            _, last_val = network.apply(train_state.params, last_obs)
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = (transition.reward + config["GAMMA"] * next_value * (1.0 - transition.done)
                             - transition.value)
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    return (gae, transition.value), gae
                (_, _), advantages = jax.lax.scan(
                    _get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch,
                    reverse=True, unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns
            advantages, targets = _calculate_gae(traj_batch, last_val)
            jax.debug.print(">> Advantage computation complete at time: {}", time.time())

            # PPO Update (minibatch updates)
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
                # Flatten from [T, N_env, ...] -> [T*N_env, ...]
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                # Shuffle and reshape into minibatches
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
            jax.debug.print(">> PPO update complete at time: {}", time.time())
            return (train_state, env_state, last_obs, rng), metric

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        jax.debug.print(">> Beginning main training loop at time: {}", time.time())
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        jax.debug.print(">> Finished main training loop at time: {}", time.time())
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ------------------------------
# Baseline: Environment Iteration Function (with print statements)
# ------------------------------
def env_iteration(rng, env, env_params, num_steps, num_envs):
    print(">> Starting environment iteration at time:", time.time())
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    print(">> Environment reset for baseline complete at time:", time.time())

    def step_fn(carry, _):
        env_state, last_obs, rng = carry
        rng, act_rng = jax.random.split(rng)
        # For a discrete env like CartPole, sample random actions
        action = jax.random.randint(act_rng, (num_envs,), 0, env.action_space(env_params).n)
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rngs, env_state, action, env_params)
        return (env_state, obsv, rng), None

    (env_state, last_obs, rng), _ = jax.lax.scan(step_fn, (env_state, obsv, rng), None, num_steps)
    print(">> Finished environment iteration at time:", time.time())
    return last_obs, env_state


# ------------------------------
# Main Benchmark Routine (with print statements)
# ------------------------------
def main():
    # Configuration (adjust TOTAL_TIMESTEPS as needed)
    config = {
        # PPO hyperparameters
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 100,   # total timesteps across all envs (use a small number for debugging)
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
        # Choose "CartPole-v1" (or "TabularMDP" if desired)
        "ENV_NAME": "TabularMDP",
        # If using TabularMDP, specify the .npz file (ignored for CartPole)
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz",
        # LR schedule flag
        "ANNEAL_LR": True,
        # Debug flag
        "DEBUG": False,
    }

    print(">> Initializing wandb at time:", time.time())
    wandb.init(project="ppo_vs_env_benchmark", config=config, reinit=True)

    # ------------------------------
    # Benchmark PPO Training
    # ------------------------------
    rng = jax.random.PRNGKey(42)
    print(">> Creating PPO train function at time:", time.time())
    ppo_train_fn = make_train(config)
    ppo_train_jit = jax.jit(ppo_train_fn)
    print(">> Starting PPO training (jitted) at time:", time.time())
    start_time = time.time()
    out = ppo_train_jit(rng)
    ppo_time = time.time() - start_time
    print("PPO training finished. Time taken: {:.4f} seconds.".format(ppo_time))

    # ------------------------------
    # Plot and log rewards from PPO training
    # ------------------------------
    metrics = jax.tree_util.tree_map(lambda x: np.array(x), out["metrics"])
    if "returned_episode_returns" in metrics:
        try:
            mean_returns = metrics["returned_episode_returns"].mean(axis=-1)
        except Exception as e:
            mean_returns = metrics["returned_episode_returns"]
        mean_returns = mean_returns.reshape(-1)
        for i, ret in enumerate(mean_returns):
            wandb.log({"update_step": i, "mean_return": ret})
        plt.figure()
        plt.plot(mean_returns, label="Mean Return")
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.title("PPO Training Performance")
        plt.legend()
        plt.tight_layout()
        wandb.log({"training_returns_plot": wandb.Image(plt)})
        plt.show()

    # ------------------------------
    # Benchmark Pure Environment Iteration
    # ------------------------------
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    total_env_steps = num_updates * config["NUM_STEPS"]

    print(">> Starting baseline environment iteration at time:", time.time())
    baseline_fn = jax.jit(lambda rng: env_iteration(rng, env, env_params, total_env_steps, config["NUM_ENVS"]))
    rng_baseline = jax.random.PRNGKey(123)
    start_time = time.time()
    _ = baseline_fn(rng_baseline)
    baseline_time = time.time() - start_time
    print("Baseline env iteration finished. Time taken: {:.4f} seconds.".format(baseline_time))

    # Log benchmark summary to wandb
    wandb.log({"ppo_time": ppo_time, "env_iteration_time": baseline_time})
    print("\n--- Benchmark Summary ---")
    print("PPO training time      : {:.4f} seconds".format(ppo_time))
    print("Env iteration time     : {:.4f} seconds".format(baseline_time))
    overhead = ppo_time - baseline_time
    print("Extra overhead (PPO)   : {:.4f} seconds".format(overhead))

    wandb.finish()

if __name__ == "__main__":
    main()
