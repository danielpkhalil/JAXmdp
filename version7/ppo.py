"""
ppo_tabular_wandb_cnn.py

- Uses PPO training for a TabularEnv or standard Gymnax env.
- Periodically performs deterministic evaluation and early stopping when the max of the deterministic evaluation
  reward and the median training reward reaches the optimal reward threshold.
- When early stopping occurs, prints the total number of environment steps taken.
- The optimal reward threshold is passed via config["OPTIMAL_REWARD"].

Usage:
    1) Ensure "gymnax_env.py" (with reward scaling) is in the same folder.
    2) pip install wandb
    3) python ppo_tabular_wandb_cnn.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb
import matplotlib.pyplot as plt

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Any, Tuple

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# Import your custom environment (with reward scaling)
from gymnax_env import TabularEnv, TabularEnvParams


# ------------------------------
# Actor-Critic Networks
# ------------------------------
class CNNActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    def setup(self):
        self.activation_fn = nn.relu if self.activation == "relu" else nn.tanh

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)

class MLPActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = act_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = act_fn(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


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
# Rollout + PPO Update Function with JIT
# ------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # 1) Rollout collection
        def env_step_fn(carry, _):
            train_state, env_state, last_obs, rng = carry
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
            return (train_state, env_state, obsv, rng), transition

        carry_init = (train_state, env_state, last_obs, rng)
        carry_out, traj_batch = jax.lax.scan(env_step_fn, carry_init, None, length=config["NUM_STEPS"])
        train_state, env_state, last_obs, rng = carry_out

        # 2) Compute GAE advantage
        _, last_val = network.apply(train_state.params, last_obs)
        def gae_scan_fn(carry, transition):
            gae, next_val = carry
            delta = transition.reward + config["GAMMA"] * next_val * (1.0 - transition.done) - transition.value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
            return (gae, transition.value), gae
        (_, _), advantages = jax.lax.scan(
            gae_scan_fn,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        returns = advantages + traj_batch.value

        # 3) PPO update with minibatch processing
        def ppo_update(train_state, traj_batch, advantages, returns, rng):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            minibatch_size = batch_size // config["NUM_MINIBATCHES"]

            # Flatten each field from [T, N_env, ...] to [batch_size, ...]
            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                traj_batch
            )
            adv_flat = advantages.reshape((batch_size,))
            ret_flat = returns.reshape((batch_size,))

            # Generate a permutation to shuffle the batch (update rng)
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)
            # Apply permutation to each leaf using tree_map
            def reshape_mb(x):
                return x.reshape((config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:])
            traj_shuf = jax.tree_util.tree_map(lambda x: reshape_mb(jnp.take(x, perm, axis=0)), traj_flat)
            adv_shuf = reshape_mb(jnp.take(adv_flat, perm, axis=0))
            ret_shuf = reshape_mb(jnp.take(ret_flat, perm, axis=0))

            def update_minibatch(train_state, minibatch):
                traj_mb, adv_mb, ret_mb = minibatch
                def loss_fn(params, t, ga, rt):
                    pi, value = network.apply(params, t.obs)
                    log_prob = pi.log_prob(t.action)
                    v_clipped = t.value + (value - t.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    v_loss_1 = (value - rt)**2
                    v_loss_2 = (v_clipped - rt)**2
                    value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))
                    ratio = jnp.exp(log_prob - t.log_prob)
                    ga_normed = (ga - ga.mean()) / (ga.std() + 1e-8)
                    pg_loss_1 = ratio * ga_normed
                    pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * ga_normed
                    policy_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
                    entropy = jnp.mean(pi.entropy())
                    loss = policy_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return loss, (policy_loss, value_loss, entropy)
                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(train_state.params, traj_mb, adv_mb, ret_mb)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss_val

            # Process each minibatch along the static first dimension
            def scan_minibatch_fn(train_state, i):
                traj_mb = jax.tree_util.tree_map(lambda x: x[i], traj_shuf)
                adv_mb = adv_shuf[i]
                ret_mb = ret_shuf[i]
                train_state, _ = update_minibatch(train_state, (traj_mb, adv_mb, ret_mb))
                return train_state, None

            indices = jnp.arange(config["NUM_MINIBATCHES"])
            train_state, _ = jax.lax.scan(scan_minibatch_fn, train_state, indices)
            return train_state, rng

        train_state, rng = ppo_update(train_state, traj_batch, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj_batch

    return rollout_and_update


# ------------------------------
# Deterministic Evaluation Function
# ------------------------------
def evaluate_policy_deterministic(train_state, network, env, env_params, rng, max_steps=10000):
    obs, state = env.reset(rng, env_params)
    done = False
    total_reward = 0.0
    steps = 0
    while (not done) and (steps < max_steps):
        pi, _ = network.apply(train_state.params, obs[None, ...])
        action = int(jnp.argmax(pi.logits[0]))
        obs, state, reward, done, info = env.step(rng, state, action, env_params)
        total_reward += float(reward)
        steps += 1
    return total_reward, steps


# ------------------------------
# Main Training Loop with Early Stopping
# ------------------------------
def run_ppo_training(config):
    # 1) Create environment with reward scaling
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)

    # 2) Build network based on observation shape
    obs_shape = env.observation_space(env_params).shape
    action_dim = env.action_space(env_params).n
    if len(obs_shape) == 3:
        network = CNNActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        network = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    # 3) Initialize network parameters and optimizer
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5)
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # 4) Reset environment
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    # 5) Build the rollout+update function
    rollout_and_update_fn = make_rollout_and_update_fn(env, env_params, network, config)

    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0

    # Rolling buffer for training episode returns
    train_returns_buffer = []
    best_eval_return = -1e9
    best_median_train = -1e9

    # Main training loop
    for update_i in range(config["NUM_UPDATES"]):
        train_state, env_state, obsv, rng, traj_batch = rollout_and_update_fn(
            train_state, env_state, obsv, rng
        )
        global_env_steps += steps_per_update

        # Extract training episode returns (assumed shape [T, N])
        info_dict = traj_batch.info
        returned_ep_ret = np.array(info_dict["returned_episode_returns"])
        returned_ep = np.array(info_dict["returned_episode"])
        ended_idx = np.where(returned_ep > 0)
        ep_returns = returned_ep_ret[ended_idx]
        for r in ep_returns:
            train_returns_buffer.append(r)

        # Deterministic evaluation every EVAL_FREQUENCY steps
        do_eval = (global_env_steps // config["EVAL_FREQUENCY"]) != ((global_env_steps - steps_per_update) // config["EVAL_FREQUENCY"])
        eval_ret = None
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, eval_steps = evaluate_policy_deterministic(train_state, network, env, env_params, eval_rng)
            best_eval_return = max(best_eval_return, eval_ret)

        # Compute median training reward from recent episodes
        N = config["TRAIN_MEDIAN_WINDOW"]
        recent_returns = train_returns_buffer[-N:] if len(train_returns_buffer) >= N else train_returns_buffer
        median_train_return = float(np.median(recent_returns)) if recent_returns else 0.0
        best_median_train = max(best_median_train, median_train_return)

        # Check early stopping: use the max of eval and median train reward
        current_eval = eval_ret if (eval_ret is not None) else -1e9
        metric_to_check = max(current_eval, median_train_return)
        wandb.log({
            "update": update_i,
            "global_env_steps": global_env_steps,
            "median_train_return": median_train_return,
            "best_median_train": best_median_train,
            "eval_return": eval_ret if eval_ret is not None else float("nan"),
            "best_eval_return": best_eval_return,
        }, step=global_env_steps)

        if metric_to_check >= config["OPTIMAL_REWARD"]:
            print(f"Optimal reward reached after {global_env_steps} env steps (metric: {metric_to_check:.3f}).")
            break

    print("Training finished.")
    return train_state


# ------------------------------
# Script Entry Point
# ------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 0,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e8,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/asterix_10_fs30/consolidated.npz",  # Path to your .npz file
        "REWARD_SCALE": 1/50,
        "EVAL_FREQUENCY": 1000,          # Evaluate every 1000 env steps
        "TRAIN_MEDIAN_WINDOW": 20,       # Window size for median training reward
        "OPTIMAL_REWARD": 8.0,          # Optimal reward threshold (after scaling)
    }
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    wandb.init(project="my_tabular_ppo_cnn_evalstop", config=config)
    run_ppo_training(config)
    wandb.finish()
