"""
Combined PPO script that:
 - Uses per-update chunking (Version 2 style)
 - vmaps over multiple seeds (Version 1 style)
 - Performs intermittent evaluation
 - Logs mean training return, median training return, best eval, etc.

Caveats:
 - Because of vmap, we can't do true early stopping per seed. All seeds run
   the same # of updates, but we do keep track if a seed *would* have stopped.
 - We do final wandb logging *after* collecting all seeds' metrics (similar
   to how Version 1 plots multi-seed results).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb

from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict, Any

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

try:
    from gymnax_env import TabularEnv, TabularEnvParams
except ImportError:
    TabularEnv, TabularEnvParams = None, None


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    """
    Three Conv layers + 512-dim FC, matching SB3's MiniGridCNN.
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2),
                    padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                    padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


class MLPActorCritic(nn.Module):
    """
    2-layer MLP fallback for non-image observations.
    """
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


# -----------------------------------------------------------------------------
# Transition Tuple
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
# Rollout + Update
# -----------------------------------------------------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    """
    Jitted function: (train_state, env_state, obs, rng) -> (new_train_state, ...)
    that does one chunk of rollout + PPO update.
    """
    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # ---------- Rollout ----------
        def env_step_fn(carry, _):
            ts, es, obs_, rng_ = carry
            rng_, act_rng = jax.random.split(rng_)
            pi, value = network.apply(ts.params, obs_)
            action = pi.sample(seed=act_rng)
            log_prob = pi.log_prob(action)

            rng_, step_rng = jax.random.split(rng_)
            step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
            obsv, es_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rngs, es, action, env_params
            )

            transition = Transition(
                done=done, action=action, value=value,
                reward=reward, log_prob=log_prob,
                obs=obs_, info=info
            )
            return (ts, es_next, obsv, rng_), transition

        carry_init = (train_state, env_state, last_obs, rng)
        (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
            env_step_fn, carry_init, xs=None, length=config["NUM_STEPS"]
        )

        # ---------- GAE Advantages ----------
        _, last_val = network.apply(train_state.params, last_obs)

        def gae_scan(carry, t: Transition):
            gae_, next_val_ = carry
            delta = t.reward + config["GAMMA"] * next_val_ * (1 - t.done) - t.value
            gae_ = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * gae_
            return (gae_, t.value), gae_

        (_, _), advantages = jax.lax.scan(
            gae_scan, (jnp.zeros_like(last_val), last_val),
            traj, reverse=True, unroll=16
        )
        returns = advantages + traj.value

        # ---------- PPO Update ----------
        def ppo_update(ts, traj_, adv_, ret_, rng_):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            mb_size = batch_size // config["NUM_MINIBATCHES"]

            # Flatten
            traj_flat = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_)
            adv_flat = adv_.reshape((batch_size,))
            ret_flat = ret_.reshape((batch_size,))

            rng_, perm_rng = jax.random.split(rng_)
            perm = jax.random.permutation(perm_rng, batch_size)

            def reshape_mb(x):
                return x.reshape((config["NUM_MINIBATCHES"], mb_size) + x.shape[1:])

            traj_shuf = jax.tree_util.tree_map(lambda x: reshape_mb(jnp.take(x, perm, axis=0)), traj_flat)
            adv_shuf = reshape_mb(jnp.take(adv_flat, perm, axis=0))
            ret_shuf = reshape_mb(jnp.take(ret_flat, perm, axis=0))

            def update_minibatch(ts_, minibatch):
                mb_traj, mb_adv, mb_ret = minibatch

                def loss_fn(params, t, ga, rt):
                    pi, val = network.apply(params, t.obs)
                    logp = pi.log_prob(t.action)

                    # Value clipping
                    v_clipped = t.value + (val - t.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    vloss1 = (val - rt) ** 2
                    vloss2 = (v_clipped - rt) ** 2
                    v_loss = 0.5 * jnp.mean(jnp.maximum(vloss1, vloss2))

                    # Policy clipping
                    ratio = jnp.exp(logp - t.log_prob)
                    ga_norm = (ga - ga.mean()) / (ga.std() + 1e-8)
                    pg1 = ratio * ga_norm
                    pg2 = jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * ga_norm
                    p_loss = -jnp.mean(jnp.minimum(pg1, pg2))

                    # Entropy
                    entropy = jnp.mean(pi.entropy())

                    # Combine
                    loss = p_loss + config["VF_COEF"] * v_loss - config["ENT_COEF"] * entropy
                    return loss, (p_loss, v_loss, entropy)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, _aux), grads = grad_fn(ts_.params, mb_traj, mb_adv, mb_ret)
                ts_ = ts_.apply_gradients(grads=grads)
                return ts_, loss_val

            def scan_minibatch(ts_, _):
                # We iterate over all minibatches once per epoch
                def scan_single_mb(ts_mb, i):
                    mb = (
                        jax.tree_util.tree_map(lambda x: x[i], traj_shuf),
                        adv_shuf[i],
                        ret_shuf[i],
                    )
                    ts_mb, _ = update_minibatch(ts_mb, mb)
                    return ts_mb, None

                idxs = jnp.arange(config["NUM_MINIBATCHES"])
                ts_, _ = jax.lax.scan(scan_single_mb, ts_, idxs)
                return ts_, None

            # Repeat for config["UPDATE_EPOCHS"]
            idxs_epoch = jnp.arange(config["UPDATE_EPOCHS"])
            ts, _ = jax.lax.scan(scan_minibatch, ts, idxs_epoch)
            return ts, rng_

        train_state, rng = ppo_update(train_state, traj, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj

    return rollout_and_update


# -----------------------------------------------------------------------------
# Deterministic Evaluation
# -----------------------------------------------------------------------------
def evaluate_policy_deterministic(train_state, network, env, env_params, rng, max_steps=10000):
    obs, state = env.reset(rng, env_params)
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < max_steps:
        pi, _val = network.apply(train_state.params, obs[None])
        action = int(jnp.argmax(pi.logits[0]))
        obs, state, rew, done, info = env.step(rng, state, action, env_params)
        total_reward += float(rew)
        steps += 1
    return total_reward, steps


# -----------------------------------------------------------------------------
# Single-seed training
# -----------------------------------------------------------------------------
def train_single_seed(rng: jax.random.PRNGKey, config: dict) -> Dict[str, jnp.ndarray]:
    """
    A single-seed training run that:
      - sets up env + network
      - runs a Python loop for config["NUM_UPDATES"]
      - calls the jitted rollout_and_update each time
      - collects metrics in JAX arrays (or host arrays)
    Since we want to vmap this function, we must avoid calling np.array(...) on
    JAX tracers *inside* a trace. Hence we use jax.tree_util.tree_map outside
    the jit calls.
    """
    # 1) Create environment
    if config["ENV_NAME"] == "TabularMDP" and TabularEnv is not None:
        env_, env_params_ = TabularEnv(config["ENV_FILE"]), None
        env_params_ = env_.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env_, env_params_ = gymnax.make(config["ENV_NAME"])

    env_ = LogWrapper(env_)

    # 2) Build network
    obs_shape = env_.observation_space(env_params_).shape
    act_dim = env_.action_space(env_params_).n
    if "MiniGrid" in config["ENV_NAME"]:
        net = MiniGridCNNActorCritic(action_dim=act_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    elif len(obs_shape) == 3:
        net = MiniGridCNNActorCritic(action_dim=act_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        net = MLPActorCritic(action_dim=act_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # 3) Initialize vector of env states
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obs, env_state = jax.vmap(env_.reset, in_axes=(0, None))(reset_rngs, env_params_)

    # 4) Build rollout+update fn
    rollout_and_update = make_rollout_and_update_fn(env_, env_params_, net, config)

    # 5) Track metrics across updates
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0

    # We'll store metrics each update in lists
    mean_train_returns = []
    median_train_returns = []
    eval_returns = []
    best_eval_returns = []
    train_returns_buffer = []
    best_eval = -1e9
    early_stop_flags = []

    # Main loop
    for update_i in range(config["NUM_UPDATES"]):
        train_state, env_state, obs, rng, traj_batch = rollout_and_update(
            train_state, env_state, obs, rng
        )
        global_env_steps += steps_per_update

        # Convert JAX tracer info -> host arrays once per update
        info_host = jax.tree_util.tree_map(lambda x: np.asarray(x), traj_batch.info)
        # shape [NUM_STEPS, NUM_ENVS]
        returned_ep_ret = info_host["returned_episode_returns"]
        returned_ep = info_host["returned_episode"]

        # (A) Mean training return for this update
        mean_ret = returned_ep_ret.mean()  # now a float
        mean_train_returns.append(mean_ret)

        # (B) For median: find which episodes ended
        ended_idx = np.where(returned_ep > 0)
        ended_returns = returned_ep_ret[ended_idx]
        for val in ended_returns:
            train_returns_buffer.append(val)

        N = config["TRAIN_MEDIAN_WINDOW"]
        recent = train_returns_buffer[-N:] if len(train_returns_buffer) >= N else train_returns_buffer
        med_train = float(np.median(recent)) if recent else 0.0
        median_train_returns.append(med_train)

        # (C) Possibly eval
        do_eval = (global_env_steps // config["EVAL_FREQUENCY"]) != (
            (global_env_steps - steps_per_update) // config["EVAL_FREQUENCY"]
        )
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, _steps = evaluate_policy_deterministic(train_state, net, env_, env_params_, eval_rng)
            best_eval = max(best_eval, eval_ret)
        else:
            eval_ret = np.nan
        eval_returns.append(eval_ret)
        best_eval_returns.append(best_eval)

        # (D) Early stop check
        metric_to_check = max(med_train, eval_ret if not np.isnan(eval_ret) else -1e9)
        early_stop_flags.append(metric_to_check >= config["OPTIMAL_REWARD"])

    # Return JAX arrays for vmap
    out = {
        "mean_train_returns": jnp.array(mean_train_returns),
        "median_train_returns": jnp.array(median_train_returns),
        "eval_returns": jnp.array(eval_returns),
        "best_eval_returns": jnp.array(best_eval_returns),
        "early_stop_flags": jnp.array(early_stop_flags),
    }
    return out


# -----------------------------------------------------------------------------
# Multi-seed driver (vmap)
# -----------------------------------------------------------------------------
def run_ppo_training_multi_seed(rng_seeds: jnp.ndarray, config: dict) -> Dict[str, jnp.ndarray]:
    """
    Vectorizes train_single_seed over multiple random seeds.
    Returns a dict of arrays with shape [num_seeds, num_updates].
    """
    return jax.vmap(train_single_seed, in_axes=(0, None))(rng_seeds, config)


# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,  # for quick testing
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
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/fruitbot_easy_l0_40_fs8/consolidated.npz",
        "REWARD_SCALE": 1.0,
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 5.0,
    }

    # Compute how many updates
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    # Initialize wandb
    wandb.init(project="combined_ppo_vmap", config=config)

    # Multiple seeds
    num_seeds = 4
    base_rng = jax.random.PRNGKey(config["SEED"])
    rng_seeds = jax.random.split(base_rng, num_seeds)

    # Multi-seed training via vmap
    results = run_ppo_training_multi_seed(rng_seeds, config)
    # e.g. results["mean_train_returns"] has shape [num_seeds, num_updates]

    # For demonstration, let's log final metrics
    mean_train_array = np.array(results["mean_train_returns"])  # shape [num_seeds, num_updates]
    median_train_array = np.array(results["median_train_returns"])
    eval_array = np.array(results["eval_returns"])
    best_eval_array = np.array(results["best_eval_returns"])
    early_stops = np.array(results["early_stop_flags"])

    # Log final update's metrics per seed
    for s in range(num_seeds):
        wandb.log({
            f"final_mean_train_return_seed{s}": float(mean_train_array[s, -1]),
            f"final_median_train_return_seed{s}": float(median_train_array[s, -1]),
            f"final_eval_return_seed{s}": float(eval_array[s, -1]),
            f"best_eval_return_seed{s}": float(best_eval_array[s, -1]),
            f"did_early_stop_seed{s}": bool(early_stops[s].max()),
        })

    # Optional: average across seeds
    final_mean_avg = float(mean_train_array[:, -1].mean())
    final_median_avg = float(median_train_array[:, -1].mean())
    final_eval_avg = float(eval_array[:, -1].mean())

    wandb.log({
        "final_mean_train_avg_across_seeds": final_mean_avg,
        "final_median_train_avg_across_seeds": final_median_avg,
        "final_eval_avg_across_seeds": final_eval_avg,
    })

    wandb.finish()
    print("Multi-seed training complete!")
