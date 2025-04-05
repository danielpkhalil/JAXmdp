import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
from typing import NamedTuple

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# Import our custom framestacking Env & Params.
try:
    from gymnax_env import TabularEnv, TabularEnvParams
except ImportError:
    TabularEnv, TabularEnvParams = None, None


# -----------------------------------------------------------------------------
# Actor-Critic Networks
# -----------------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    """
    CNN architecture replicating SB3's MiniGridCNN but with explicit downsampling.
    For an input of shape (84,84, 3*num_frames) (e.g. (84,84,12) for 4 stacked frames),
    the layers are:
      - Conv1: kernel_size=(3,3), strides=(2,2)   -> output: (42,42,32)
      - Conv2: kernel_size=(3,3), strides=(1,1)   -> output: (42,42,64)
      - MaxPool: window_shape=(2,2), strides=(2,2)  -> output: (21,21,64)
      - Conv3: kernel_size=(3,3), strides=(1,1)   -> output: (21,21,64)
    Flattening gives 21×21×64 = 28224.
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        # Conv1: downsample spatially by factor 2.
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2),
                    padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        # Conv2: no downsampling.
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        # MaxPool: explicitly downsample spatially.
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        # Conv3: process features (no additional downsampling).
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        # Flatten the spatial dimensions.
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


class MLPActorCritic(nn.Module):
    """
    Simple 2x64 MLP for non-image observations.
    """
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act_fn(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


# -----------------------------------------------------------------------------
# Transition NamedTuple
# -----------------------------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# -----------------------------------------------------------------------------
# Main make_train(config) => train(rng) function
# -----------------------------------------------------------------------------
def make_train(config):
    """
    Returns a function `train(rng)` that:
      - Initializes the environment and the policy network.
      - Runs a fully-jitted PPO loop for NUM_UPDATES.
      - Returns a dict with the final runner state and collected metrics.

    This function passes the `num_frames` parameter to the environment so that it
    returns stacked observations.
    """
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    if config["ENV_NAME"] == "TabularMDP" and TabularEnv is not None:
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(
            reward_scale=config.get("REWARD_SCALE", 1.0),
            num_frames=config.get("NUM_FRAMES", 1),
        )
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        obs_shape = env.observation_space(env_params).shape
        action_dim = env.action_space(env_params).n

        if len(obs_shape) == 3:
            network = MiniGridCNNActorCritic(action_dim=action_dim)
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
        else:
            network = MLPActorCritic(action_dim=action_dim, activation=config.get("ACTIVATION", "relu"))
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

        rng, init_rng = jax.random.split(rng)
        network_params = network.init(init_rng, dummy_obs)

        if config.get("ANNEAL_LR", False):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5)
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5)
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        def _update_step(runner_state, update_idx):
            train_state, env_state, last_obs, rng, global_step_count = runner_state

            def _env_step(carry, _):
                ts, st, obs, key = carry
                key, key_act = jax.random.split(key)
                pi, value = network.apply(ts.params, obs)
                action = pi.sample(seed=key_act)
                log_prob = pi.log_prob(action)
                key, key_step = jax.random.split(key)
                step_keys = jax.random.split(key_step, config["NUM_ENVS"])
                obsv_next, st_next, rew, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, st, action, env_params)
                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=rew,
                    log_prob=log_prob,
                    obs=obs,
                    info=info,
                )
                return (ts, st_next, obsv_next, key), transition

            carry_init = (train_state, env_state, last_obs, rng)
            (train_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, carry_init, None, config["NUM_STEPS"]
            )

            _, last_val = network.apply(train_state.params, last_obs)

            def _gae_scan(carry, t: Transition):
                gae, next_val = carry
                delta = t.reward + config["GAMMA"] * next_val * (1 - t.done) - t.value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * gae
                return (gae, t.value), gae

            (_, _), advantages = jax.lax.scan(
                _gae_scan,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16
            )
            returns = advantages + traj_batch.value

            def _update_epoch(update_state, _):
                def _update_minibatch(ts, minibatch):
                    mb_traj, mb_adv, mb_ret = minibatch

                    def loss_fn(params, t, adv_, rt_):
                        pi, val = network.apply(params, t.obs)
                        logp = pi.log_prob(t.action)
                        v_clipped = t.value + (val - t.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        v_loss_1 = (val - rt_) ** 2
                        v_loss_2 = (v_clipped - rt_) ** 2
                        value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))
                        ratio = jnp.exp(logp - t.log_prob)
                        adv_norm = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
                        pg_loss_1 = ratio * adv_norm
                        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                                             1.0 + config["CLIP_EPS"]) * adv_norm
                        policy_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
                        entropy = jnp.mean(pi.entropy())
                        loss = policy_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return loss, (policy_loss, value_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss_val, _aux), grads = grad_fn(ts.params, mb_traj, mb_adv, mb_ret)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, loss_val

                ts, traj_b, adv_b, ret_b, k_ = update_state
                k_, pk = jax.random.split(k_)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                perm = jax.random.permutation(pk, batch_size)

                def flatten(x):
                    return x.reshape((batch_size,) + x.shape[2:])

                traj_flat = jax.tree_util.tree_map(flatten, traj_b)
                adv_flat = adv_b.reshape((batch_size,))
                ret_flat = ret_b.reshape((batch_size,))

                def minibatchify(x):
                    mbsize = batch_size // config["NUM_MINIBATCHES"]
                    x_ = jnp.take(x, perm, axis=0)
                    return x_.reshape((config["NUM_MINIBATCHES"], mbsize) + x_.shape[1:])

                traj_mb = jax.tree_util.tree_map(minibatchify, traj_flat)
                adv_mb = minibatchify(adv_flat)
                ret_mb = minibatchify(ret_flat)

                def scan_minibatch(ts, i):
                    batch_i = (
                        jax.tree_util.tree_map(lambda xx: xx[i], traj_mb),
                        adv_mb[i],
                        ret_mb[i],
                    )
                    ts, _ = _update_minibatch(ts, batch_i)
                    return ts, None

                for _ in range(config["UPDATE_EPOCHS"]):
                    indices = jnp.arange(config["NUM_MINIBATCHES"])
                    ts, _ = jax.lax.scan(scan_minibatch, ts, indices)
                new_state = (ts, traj_b, adv_b, ret_b, k_)
                return new_state, None

            update_state = (train_state, traj_batch, advantages, returns, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, length=1)
            train_state = update_state[0]
            rng = update_state[-1]
            metrics = traj_batch.info
            global_step_count += (config["NUM_ENVS"] * config["NUM_STEPS"])
            new_runner_state = (train_state, env_state, last_obs, rng, global_step_count)
            return new_runner_state, metrics

        runner_state = (train_state, env_state, obsv, rng, jnp.array(0))
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config["NUM_UPDATES"]))
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# -----------------------------------------------------------------------------
# Script Entry Point with wandb Logging
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import wandb

    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps_with_exploration_policy/breakout_10_fs30/consolidated_framestack.npz",
        "REWARD_SCALE": 1.0,
        "NUM_FRAMES": 4,
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 5.0,
        "ANNEAL_LR": True,
    }

    wandb.init(project="ppo_framestack_example", config=config)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    elapsed_single = time.time() - t0
    print(f"Single-seed training took {elapsed_single:.2f} seconds.")
    returned_ep_ret = out["metrics"]["returned_episode_returns"]
    mean_return_per_update = returned_ep_ret.mean(axis=(-1, -2))
    plt.figure()
    plt.plot(mean_return_per_update, label="Single Seed")
    plt.xlabel("Update")
    plt.ylabel("Mean Episode Return")
    plt.title("Single-Seed PPO on TabularMDP with Framestacking")
    plt.legend()
    wandb.log({"training_returns_plot_single_seed": wandb.Image(plt)})
    plt.close()
    num_seeds = 4
    rng_seeds = jax.random.split(jax.random.PRNGKey(999), num_seeds)
    batched_train = jax.jit(jax.vmap(train_fn))
    t0 = time.time()
    outs = jax.block_until_ready(batched_train(rng_seeds))
    elapsed_multi = time.time() - t0
    print(f"{num_seeds}-seed training took {elapsed_multi:.2f} seconds.")
    rets_all = outs["metrics"]["returned_episode_returns"]
    plt.figure()
    for i in range(num_seeds):
        mean_ret_i = rets_all[i].mean(axis=(-1, -2))
        plt.plot(mean_ret_i, label=f"seed {i}")
    plt.xlabel("Update")
    plt.ylabel("Mean Episode Return")
    plt.title("Multi-Seed PPO on TabularMDP with Framestacking")
    plt.legend()
    wandb.log({"training_returns_plot_multi_seed": wandb.Image(plt)})
    plt.close()
    wandb.log({
        "final_single_seed_return": float(mean_return_per_update[-1]),
        "time_single_seed": elapsed_single,
        "time_multi_seed_for_4": elapsed_multi,
    })
    wandb.finish()
