"""
ppo_minigrid_wandb_cnn.py

Example JAX PPO script modified to match the SB3 "MiniGridCNN" architecture exactly.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Any, Tuple

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# If you have a custom TabularEnv, etc.
from gymnax_env import TabularEnv, TabularEnvParams


# ------------------------------
# MiniGridCNN Actor-Critic
# ------------------------------
class MiniGridCNNActorCritic(nn.Module):
    """
    Replicates the three Conv layers plus a 512-dim FC layer from the SB3 MiniGridCNN:
    1) Conv2D( in->32, kernel=3, stride=2, padding=1 ) + ReLU
    2) Conv2D(32->64, kernel=3, stride=2, padding=1 ) + ReLU
    3) Conv2D(64->64, kernel=3, stride=1, padding=1 ) + ReLU
    4) Flatten
    5) Dense(512) + ReLU
    Then separate outputs for action logits and value.
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x expected shape: (batch_size, H, W, C)
        # No /255.0 normalization here to match SB3's MiniGridCNN defaults.
        # Cast input to float32 even if not normalizing by 255.
        x = x.astype(jnp.float32)
        # Convolution layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # 512-dim FC
        x = nn.Dense(features=512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Policy head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        # Value head
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


# ------------------------------
# MLP Actor-Critic (unchanged)
# ------------------------------
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
# Rollout + PPO Update Function
# ------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # 1) Collect rollout
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

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                info=info,
            )
            return (train_state, env_state, obsv, rng), transition

        carry_init = (train_state, env_state, last_obs, rng)
        carry_out, traj_batch = jax.lax.scan(
            env_step_fn, carry_init, None, length=config["NUM_STEPS"]
        )
        train_state, env_state, last_obs, rng = carry_out

        # 2) GAE advantage
        _, last_val = network.apply(train_state.params, last_obs)

        def gae_scan_fn(carry, transition):
            gae, next_val = carry
            delta = (
                transition.reward
                + config["GAMMA"] * next_val * (1.0 - transition.done)
                - transition.value
            )
            gae = (
                delta
                + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
            )
            return (gae, transition.value), gae

        (_, _), advantages = jax.lax.scan(
            gae_scan_fn,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        returns = advantages + traj_batch.value

        # 3) PPO update w/ minibatching
        def ppo_update(train_state, traj_batch, advantages, returns, rng):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            minibatch_size = batch_size // config["NUM_MINIBATCHES"]

            # Flatten [T, N_env, ...] -> [batch_size, ...]
            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                traj_batch,
            )
            adv_flat = advantages.reshape((batch_size,))
            ret_flat = returns.reshape((batch_size,))

            # Shuffle
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)

            def reshape_mb(x):
                return x.reshape(
                    (config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:]
                )

            traj_shuf = jax.tree_util.tree_map(
                lambda x: reshape_mb(jnp.take(x, perm, axis=0)), traj_flat
            )
            adv_shuf = reshape_mb(jnp.take(adv_flat, perm, axis=0))
            ret_shuf = reshape_mb(jnp.take(ret_flat, perm, axis=0))

            def update_minibatch(train_state, minibatch):
                traj_mb, adv_mb, ret_mb = minibatch

                def loss_fn(params, t, ga, rt):
                    pi, value = network.apply(params, t.obs)
                    log_prob = pi.log_prob(t.action)

                    # Value clipping
                    v_clipped = t.value + (value - t.value).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    v_loss_1 = (value - rt) ** 2
                    v_loss_2 = (v_clipped - rt) ** 2
                    value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

                    # Policy clipping
                    ratio = jnp.exp(log_prob - t.log_prob)
                    ga_normed = (ga - ga.mean()) / (ga.std() + 1e-8)
                    pg_loss_1 = ratio * ga_normed
                    pg_loss_2 = jnp.clip(
                        ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                    ) * ga_normed
                    policy_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                    # Entropy
                    entropy = jnp.mean(pi.entropy())

                    # Final loss
                    loss = (
                        policy_loss
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return loss, (policy_loss, value_loss, entropy)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(train_state.params, traj_mb, adv_mb, ret_mb)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss_val

            def scan_minibatch_fn(train_state, i):
                traj_mb = jax.tree_util.tree_map(lambda x: x[i], traj_shuf)
                adv_mb = adv_shuf[i]
                ret_mb = ret_shuf[i]
                train_state, _ = update_minibatch(train_state, (traj_mb, adv_mb, ret_mb))
                return train_state, None

            # Run multiple epochs
            # By default, SB3 uses "n_epochs=4" => that means we do 4 passes over these minibatches.
            # The snippet below just does 1 pass. So if you want exactly 4, add a `for` or lax.scan for 4 passes:
            for _ in range(config["UPDATE_EPOCHS"]):
                indices = jnp.arange(config["NUM_MINIBATCHES"])
                train_state, _ = jax.lax.scan(scan_minibatch_fn, train_state, indices)

            return train_state, rng

        train_state, rng = ppo_update(train_state, traj_batch, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj_batch

    return rollout_and_update


# ------------------------------
# Deterministic Evaluation
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
# Main Training Loop
# ------------------------------
def run_ppo_training(config):
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    env = LogWrapper(env)

    # 2) Build the correct network
    obs_shape = env.observation_space(env_params).shape
    action_dim = env.action_space(env_params).n
    if "MiniGrid" in config["ENV_NAME"]:
        # Use the specialized MiniGridCNN architecture
        network = MiniGridCNNActorCritic(action_dim=action_dim)
        # Observations in MiniGrid are typically uint8 images [H, W, 3]
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    elif len(obs_shape) == 3:
        # If it's some other 2D image-based environment
        # or your default CNN for something else
        network = MiniGridCNNActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        # MLP fallback
        network = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    # 3) Initialize parameters & optimizer
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # 4) Reset vector of environments
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    # 5) Build rollout + update
    rollout_and_update_fn = make_rollout_and_update_fn(env, env_params, network, config)

    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0

    # Track training returns
    train_returns_buffer = []
    best_eval_return = -1e9
    best_median_train = -1e9

    # Main loop
    for update_i in range(config["NUM_UPDATES"]):
        train_state, env_state, obsv, rng, traj_batch = rollout_and_update_fn(
            train_state, env_state, obsv, rng
        )
        global_env_steps += steps_per_update

        # Episode returns from info
        info_dict = traj_batch.info
        returned_ep_ret = np.array(info_dict["returned_episode_returns"])
        returned_ep = np.array(info_dict["returned_episode"])
        ended_idx = np.where(returned_ep > 0)
        ep_returns = returned_ep_ret[ended_idx]
        for r in ep_returns:
            train_returns_buffer.append(r)

        # Evaluate periodically
        do_eval = (global_env_steps // config["EVAL_FREQUENCY"]) != (
            (global_env_steps - steps_per_update) // config["EVAL_FREQUENCY"]
        )
        eval_ret = None
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, eval_steps = evaluate_policy_deterministic(
                train_state, network, env, env_params, eval_rng
            )
            best_eval_return = max(best_eval_return, eval_ret)

        # Median training reward
        N = config["TRAIN_MEDIAN_WINDOW"]
        recent_returns = train_returns_buffer[-N:] if len(train_returns_buffer) >= N else train_returns_buffer
        median_train_return = float(np.median(recent_returns)) if recent_returns else 0.0
        best_median_train = max(best_median_train, median_train_return)

        # Log to W&B
        metric_to_check = max(eval_ret if eval_ret else -1e9, median_train_return)
        wandb.log(
            {
                "update": update_i,
                "global_env_steps": global_env_steps,
                "median_train_return": median_train_return,
                "best_median_train": best_median_train,
                "eval_return": eval_ret if eval_ret is not None else float("nan"),
                "best_eval_return": best_eval_return,
            },
            step=global_env_steps,
        )

        # Early stopping
        if metric_to_check >= config["OPTIMAL_REWARD"]:
            print(
                f"Optimal reward reached after {global_env_steps} env steps "
                f"(metric: {metric_to_check:.3f})."
            )
            break

    print("Training finished.")
    return train_state


# ------------------------------
# Script Entry Point
# ------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,  # Matches SB3's typical PPO n_epochs
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "TabularMDP",  # Example MiniGrid name
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-Unlock-v0/consolidated.npz",  # Not used unless you do "TabularMDP"
        "REWARD_SCALE": 1.0,
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 1.0,  # Example
    }
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    wandb.init(project="my_minigrid_ppo_cnn_evalstop", config=config)
    run_ppo_training(config)
    wandb.finish()
