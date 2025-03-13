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
    - Allows multi-seed training in parallel using jax.vmap.
    - Logs some final aggregated results to Weights & Biases
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
        # x shape: (batch, H, W, C), uint8 in [0..255]
        # We'll normalize by 255.0, do a couple of conv layers, then flatten and produce heads.
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
    """
    Actor-critic model for 1D (non-image) discrete action spaces.
    """
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        # Flatten if needed
        x = x.reshape((x.shape[0], -1))

        # Policy
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation_fn(actor)
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation_fn(actor)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
        pi = distrax.Categorical(logits=actor_logits)

        # Value function
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation_fn(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation_fn(critic)
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
# make_train(config): returns a function that trains PPO
# on a *single* PRNG key/seed. Then we can vmap it across seeds.
# ------------------------------
def make_train(config):
    """
    Builds a single-seed training function for PPO. We can then vmap this
    function over a batch of PRNG keys to get multi-seed training in parallel.
    """

    # Precompute
    num_updates = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    minibatch_size = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # 1) Create environment
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    # Log wrapper
    env = LogWrapper(env)

    # 2) Learning rate schedule
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    # 3) Build the training function
    def train_one_seed(rng: jnp.ndarray):
        """
        Runs PPO for a single seed RNG.
        Returns a dictionary of final metrics, including the entire metrics log.
        """
        # Decide if we want CNN or MLP
        obs_shape = env.observation_space(env_params).shape
        action_dim = env.action_space(env_params).n

        if len(obs_shape) == 3:
            # Possibly images => CNN
            network_def = CNNActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
        else:
            # 1D => MLP
            network_def = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

        rng, init_rng = jax.random.split(rng)
        params = network_def.init(init_rng, dummy_obs)

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
            apply_fn=network_def.apply,
            params=params,
            tx=tx,
        )

        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # -----------
        # define a single "update_step" that does:
        #   1) collect rollout
        #   2) compute GAE advantage
        #   3) run PPO training
        # -----------
        def update_step(carry, unused):
            train_state_, env_state_, last_obs_, rng_ = carry

            # 1) Rollout
            def rollout_step(carry, unused):
                ts, es, lo, rng__ = carry
                rng__, act_rng = jax.random.split(rng__)
                pi, val = network_def.apply(ts.params, lo)
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)

                rng__, step_rng = jax.random.split(rng__)
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                next_obs, next_es, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    step_rngs, es, action, env_params
                )
                transition = Transition(done, action, val, reward, log_prob, lo, info)
                carry_out = (ts, next_es, next_obs, rng__)
                return carry_out, transition

            carry_out, traj_batch = jax.lax.scan(
                rollout_step, (train_state_, env_state_, last_obs_, rng_), None, config["NUM_STEPS"]
            )
            train_state_, env_state_, last_obs_, rng_ = carry_out

            # 2) GAE advantage
            _, last_val = network_def.apply(train_state_.params, last_obs_)

            def gae_scan_fn(carry2, transition):
                gae_, next_val_ = carry2
                delta = (
                    transition.reward
                    + config["GAMMA"] * next_val_ * (1.0 - transition.done)
                    - transition.value
                )
                gae_ = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae_
                return (gae_, transition.value), gae_

            (_, _), advantages = jax.lax.scan(
                gae_scan_fn, (jnp.zeros_like(last_val), last_val), traj_batch,
                reverse=True, unroll=16
            )
            returns = advantages + traj_batch.value

            # 3) PPO update over multiple epochs
            def ppo_epochs(carry3, _):
                ts_, rng__ = carry3

                # Shuffle
                rng__, perm_rng = jax.random.split(rng__)
                batch_size_ = config["NUM_STEPS"] * config["NUM_ENVS"]
                perm = jax.random.permutation(perm_rng, batch_size_)

                def flatten_and_shuffle(x):
                    # Flatten from [T, N_env, ...] -> [batch_size, ...]
                    x = x.reshape((batch_size_,) + x.shape[2:])
                    return jnp.take(x, perm, axis=0)

                traj_flat = jax.tree_util.tree_map(flatten_and_shuffle, traj_batch)
                adv_flat = flatten_and_shuffle(advantages)
                ret_flat = flatten_and_shuffle(returns)

                # Minibatches
                def reshape_mb(x):
                    return x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:])

                traj_mb = jax.tree_util.tree_map(reshape_mb, traj_flat)
                adv_mb = reshape_mb(adv_flat)
                ret_mb = reshape_mb(ret_flat)

                # Iterate over minibatches
                def update_minibatch(ts__, mini_idx):
                    # gather the i-th minibatch
                    tb_i = jax.tree_util.tree_map(lambda x: x[mini_idx], traj_mb)
                    ad_i = adv_mb[mini_idx]
                    rt_i = ret_mb[mini_idx]

                    def loss_fn(params, trans, gae, ret_):
                        pi_, val_ = network_def.apply(params, trans.obs)
                        logp_ = pi_.log_prob(trans.action)

                        # Value loss
                        v_clipped = trans.value + (val_ - trans.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        v_loss_1 = (val_ - ret_)**2
                        v_loss_2 = (v_clipped - ret_)**2
                        value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

                        # Policy loss
                        ratio = jnp.exp(logp_ - trans.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm
                        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
                        policy_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                        # Entropy
                        entropy = jnp.mean(pi_.entropy())

                        total_loss = (
                            policy_loss
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (policy_loss, value_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss_val, _aux), grads = grad_fn(ts__.params, tb_i, ad_i, rt_i)
                    ts__ = ts__.apply_gradients(grads=grads)
                    return ts__, None

                def scan_mb(ts__, i_):
                    return update_minibatch(ts__, i_)

                ts_, _ = jax.lax.scan(scan_mb, ts_, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts_, rng__), None

            ppo_state = (train_state_, rng_)
            ppo_state, _ = jax.lax.scan(ppo_epochs, ppo_state, None, config["UPDATE_EPOCHS"])
            train_state_, rng_ = ppo_state

            runner_state_new = (train_state_, env_state_, last_obs_, rng_)
            # For logging, we return the "info" from the final rollout batch
            return runner_state_new, traj_batch.info

        # We'll do a single big scan over the number of updates
        runner_state_init = (train_state, env_state, obsv, rng)
        (train_state_final, env_state_final, obs_final, rng_final), metrics = jax.lax.scan(
            update_step, runner_state_init, None, length=num_updates
        )

        # Return final data
        return {
            "runner_state": {
                "train_state": train_state_final,
                "env_state": env_state_final,
                "obs": obs_final,
                "rng": rng_final,
            },
            "metrics": metrics,
        }

    return train_one_seed


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    config = {
        # PPO hyperparams
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e6,
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

        # ENV setup
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "breakout_10_fs30.npz",
        "REWARD_SCALE": 1.0,   # scale factor on environment rewards

        # LR schedule
        "ANNEAL_LR": True,
    }

    # 1) Init W&B
    wandb.init(project="my_tabular_ppo_cnn_vmap", config=config)

    # 2) Build the single-seed PPO function
    train_single_fn = make_train(config)

    # 3) JIT compile
    train_single_jit = jax.jit(train_single_fn)

    # 4) Multi-seed in parallel via jax.vmap
    num_seeds = 5
    base_rng = jax.random.PRNGKey(0)
    rng_seeds = jax.random.split(base_rng, num_seeds)  # shape (num_seeds, 2)

    # vmap the single-seed training: shape (num_seeds,) -> each returns a dict
    all_results = jax.vmap(train_single_jit)(rng_seeds)

    # Move results to CPU (host)
    all_results = jax.tree_util.tree_map(lambda x: np.asarray(x), all_results)

    # For example, you might want to gather the final episodic returns
    # from the "metrics" field of each seed. The "metrics" is shape
    # [num_updates, num_steps, num_envs, ...], or something similar, depending on what LogWrapper logs.
    # We'll just do a naive example:
    final_info_per_seed = all_results["metrics"]["returned_episode_returns"]

    # Let's do a trivial metric: average final returned_episode_returns across seeds
    # final_info_per_seed: shape [num_seeds, num_updates, num_steps, num_envs] (if it logs every step)
    # We'll just take the mean over the last dimension for demonstration:
    mean_returns_across_seeds = np.mean(final_info_per_seed, axis=(0, 2, 3))  # shape [num_seeds, num_updates] => hopefully

    # Just as a demonstration, let's log the final mean across seeds:
    final_return_value = float(np.mean(mean_returns_across_seeds[-1]))  # average final update across seeds
    wandb.log({"final_mean_return_across_seeds": final_return_value})

    # We can also do a quick plot
    plt.figure()
    for seed_i in range(num_seeds):
        # shape [num_updates, num_steps, num_envs], do a per-update mean:
        single_seed_data = final_info_per_seed[seed_i]  # shape [num_updates, ...]
        mean_across_steps_envs = single_seed_data.mean(axis=(1,2))  # shape [num_updates]
        plt.plot(mean_across_steps_envs, label=f"Seed {seed_i}")
    plt.title("PPO Training - Multi-Seed Average Returns")
    plt.xlabel("Training Update")
    plt.ylabel("Mean Episode Return")
    plt.legend()
    plt.tight_layout()
    wandb.log({"multi_seed_plot": wandb.Image(plt)})
    plt.show()

    wandb.finish()
    print("Multi-seed training complete!")
