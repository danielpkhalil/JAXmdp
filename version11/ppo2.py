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

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Tuple, Dict, Any

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# If you have a custom TabularEnv, etc.
try:
    from gymnax_env import TabularEnv, TabularEnvParams
except ImportError:
    TabularEnv, TabularEnvParams = None, None


# -----------------------------------------------------------------------------
# Actor-Critic Networks
# -----------------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    """
    Replicates SB3's MiniGridCNN architecture (3xConv + 512-dim FC).
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Observations in MiniGrid are typically uint8 [H, W, C].
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

        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Policy
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        # Value
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


class MLPActorCritic(nn.Module):
    """
    2-layer MLP fallback if obs are not images.
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
# Build the "rollout_and_update" step (chunked approach)
# -----------------------------------------------------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    """
    Returns a jitted function that:
     1) Collects NUM_STEPS of data from a vector of envs,
     2) Calculates GAE advantages,
     3) Does a PPO update with minibatching + multiple epochs.
    """

    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # ------------------ 1) Rollout ------------------
        def env_step_fn(carry, _unused):
            train_state_, env_state_, last_obs_, rng_ = carry
            rng_, act_rng = jax.random.split(rng_)
            pi, value = network.apply(train_state_.params, last_obs_)
            action = pi.sample(seed=act_rng)
            log_prob = pi.log_prob(action)

            rng_, step_rng = jax.random.split(rng_)
            step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
            obsv, env_state_, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_rngs, env_state_, action, env_params)

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs_,
                info=info,
            )
            return (train_state_, env_state_, obsv, rng_), transition

        carry_init = (train_state, env_state, last_obs, rng)
        (train_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(
            env_step_fn, carry_init, None, length=config["NUM_STEPS"]
        )

        # ------------------ 2) GAE advantage ------------------
        _, last_val = network.apply(train_state.params, last_obs)

        def gae_scan_fn(carry, transition):
            gae_, next_val_ = carry
            delta = (
                transition.reward
                + config["GAMMA"] * next_val_ * (1.0 - transition.done)
                - transition.value
            )
            gae_ = (
                delta
                + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae_
            )
            return (gae_, transition.value), gae_

        (_, _), advantages = jax.lax.scan(
            gae_scan_fn,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        returns = advantages + traj_batch.value

        # ------------------ 3) PPO update w/ minibatching ------------------
        def ppo_update(train_state_, traj_batch_, advantages_, returns_, rng_):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            minibatch_size = batch_size // config["NUM_MINIBATCHES"]

            # Flatten [T, N_env, ...] -> [batch_size, ...]
            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch_
            )
            adv_flat = advantages_.reshape((batch_size,))
            ret_flat = returns_.reshape((batch_size,))

            # Shuffle
            rng_, perm_rng = jax.random.split(rng_)
            perm = jax.random.permutation(perm_rng, batch_size)

            def reshape_mb(x):
                return x.reshape((config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:])

            traj_shuf = jax.tree_util.tree_map(
                lambda x: reshape_mb(jnp.take(x, perm, axis=0)), traj_flat
            )
            adv_shuf = reshape_mb(jnp.take(adv_flat, perm, axis=0))
            ret_shuf = reshape_mb(jnp.take(ret_flat, perm, axis=0))

            def update_minibatch(train_state_, minibatch):
                mb_traj, mb_adv, mb_ret = minibatch

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

                    # Final combined loss
                    loss = (
                        policy_loss
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return loss, (policy_loss, value_loss, entropy)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(train_state_.params, mb_traj, mb_adv, mb_ret)
                train_state_ = train_state_.apply_gradients(grads=grads)
                return train_state_, loss_val

            def scan_minibatch_fn(train_state_, i):
                mb_traj = jax.tree_util.tree_map(lambda x: x[i], traj_shuf)
                mb_adv = adv_shuf[i]
                mb_ret = ret_shuf[i]
                train_state_, _ = update_minibatch(train_state_, (mb_traj, mb_adv, mb_ret))
                return train_state_, None

            # Repeat for config["UPDATE_EPOCHS"]
            for _ in range(config["UPDATE_EPOCHS"]):
                indices = jnp.arange(config["NUM_MINIBATCHES"])
                train_state_, _ = jax.lax.scan(scan_minibatch_fn, train_state_, indices)

            return train_state_, rng_

        train_state, rng = ppo_update(train_state, traj_batch, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj_batch

    return rollout_and_update


# -----------------------------------------------------------------------------
# Deterministic Evaluation
# -----------------------------------------------------------------------------
def evaluate_policy_deterministic(train_state, network, env, env_params, rng,
                                  max_steps=10000):
    """
    Run a single rollout (greedy w.r.t. policy logits) until done or max_steps.
    Returns total_reward, steps_taken.
    """
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


# -----------------------------------------------------------------------------
# Single-seed Training with a Python Loop (chunked updates)
# -----------------------------------------------------------------------------
def train_single_seed(rng: jax.random.PRNGKey, config: dict) -> Dict[str, jnp.ndarray]:
    """
    Runs the entire training loop (chunked by NUM_STEPS) for a single seed.
    Returns a dict of *arrays* of metrics (one entry per update), so that we can
    vmap over multiple seeds if desired.

    We do NOT do wandb logging inside this function, because that won't work
    under vmap. Instead we return the arrays for later analysis/logging.
    """
    # 1) Environment creation
    if config["ENV_NAME"] == "TabularMDP" and TabularEnv is not None:
        env_ = TabularEnv(config["ENV_FILE"])
        env_params_ = env_.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env_, env_params_ = gymnax.make(config["ENV_NAME"])

    env_ = LogWrapper(env_)

    # 2) Construct the network
    obs_shape = env_.observation_space(env_params_).shape
    action_dim = env_.action_space(env_params_).n
    if "MiniGrid" in config["ENV_NAME"]:
        network = MiniGridCNNActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    elif len(obs_shape) == 3:
        network = MiniGridCNNActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        network = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    # 3) Initialize params & optimizer
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # 4) Vector of env states
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env_.reset, in_axes=(0, None))(reset_rngs, env_params_)

    # 5) Build the rollout+update function
    rollout_and_update_fn = make_rollout_and_update_fn(env_, env_params_, network, config)

    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0

    # We'll track a buffer of episode returns (to compute median of last N).
    train_returns_buffer = []

    # We'll store metrics for each update in arrays/lists, then convert to jnp.
    mean_train_returns = []
    median_train_returns = []
    eval_returns = []
    best_eval_returns = []
    # Track the best eval we've seen so far
    best_eval_return = -1e9
    # Also track if we *would* have early-stopped
    early_stop_flags = []

    # MAIN TRAIN LOOP (Python)
    for update_i in range(config["NUM_UPDATES"]):
        # Rollout + PPO update
        train_state, env_state, obsv, rng, traj_batch = rollout_and_update_fn(
            train_state, env_state, obsv, rng
        )
        global_env_steps += steps_per_update

        # (A) Compute "mean training return" from this rollout
        info_dict = traj_batch.info  # shape [NUM_STEPS, NUM_ENVS]
        returned_ep_ret = np.array(info_dict["returned_episode_returns"])
        # Mean over all steps & envs
        mean_return_for_update = returned_ep_ret.mean()
        mean_train_returns.append(mean_return_for_update)

        # (B) Update train_returns_buffer for median
        returned_ep = np.array(info_dict["returned_episode"])
        ended_idx = np.where(returned_ep > 0)
        ep_returns = returned_ep_ret[ended_idx]  # returns for episodes that ended
        for r in ep_returns:
            train_returns_buffer.append(r)

        # Compute median of last N returns
        N = config["TRAIN_MEDIAN_WINDOW"]
        recent_returns = train_returns_buffer[-N:] if len(train_returns_buffer) >= N else train_returns_buffer
        median_train_return = float(np.median(recent_returns)) if recent_returns else 0.0
        median_train_returns.append(median_train_return)

        # (C) Intermittent evaluation
        do_eval = (global_env_steps // config["EVAL_FREQUENCY"]) != (
            (global_env_steps - steps_per_update) // config["EVAL_FREQUENCY"]
        )
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, _eval_steps = evaluate_policy_deterministic(
                train_state, network, env_, env_params_, eval_rng
            )
            best_eval_return = max(best_eval_return, eval_ret)
        else:
            eval_ret = np.nan  # No eval this step
        eval_returns.append(eval_ret)
        best_eval_returns.append(best_eval_return)

        # (D) Early stopping check
        metric_to_check = max(eval_ret if not np.isnan(eval_ret) else -1e9,
                              median_train_return)
        if metric_to_check >= config["OPTIMAL_REWARD"]:
            # Mark that we "would have stopped" if we could
            early_stop_flags.append(True)
        else:
            early_stop_flags.append(False)

    # Convert lists to jnp for a clean vmap-compatible return
    metrics_out = {
        "mean_train_returns": jnp.array(mean_train_returns),
        "median_train_returns": jnp.array(median_train_returns),
        "eval_returns": jnp.array(eval_returns),
        "best_eval_returns": jnp.array(best_eval_returns),
        "early_stop_flags": jnp.array(early_stop_flags),
    }
    return metrics_out


# -----------------------------------------------------------------------------
# Multi-seed PPO using vmap
# -----------------------------------------------------------------------------
def run_ppo_training_multi_seed(rng_seeds: jnp.ndarray, config: dict) -> Dict[str, jnp.ndarray]:
    """
    Runs `train_single_seed` in parallel for multiple seeds via vmap.
    rng_seeds shape: (num_seeds,)
    Returns a dict of stacked metrics, each shape: (num_seeds, NUM_UPDATES).
    """
    # We vmap over the first argument (the RNG key), while config is shared.
    batched_results = jax.vmap(train_single_seed, in_axes=(0, None))(rng_seeds, config)
    # batched_results is a dict of arrays, each shaped (num_seeds, NUM_UPDATES).
    return batched_results


# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,  # reduce for a quick demo
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
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/fruitbot_easy_l0_40_fs8/consolidated.npz",  # used if ENV_NAME == "TabularMDP"
        "REWARD_SCALE": 1.0,
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 5.0,
    }

    # Calculate number of updates
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    # Initialize wandb
    wandb.init(project="combined_ppo_vmap", config=config)

    # Create multiple seeds
    num_seeds = 4
    base_rng = jax.random.PRNGKey(config["SEED"])
    rng_seeds = jax.random.split(base_rng, num_seeds)  # shape (4,)

    # Run multi-seed training
    results = run_ppo_training_multi_seed(rng_seeds, config)
    # results is a dict of shape:
    #   {
    #       "mean_train_returns": (num_seeds, NUM_UPDATES),
    #       "median_train_returns": (num_seeds, NUM_UPDATES),
    #       "eval_returns": (num_seeds, NUM_UPDATES),
    #       ...
    #   }

    # For demonstration, let's log the final returns
    # We'll log *one line per seed* + a mean or something
    mean_train_returns = np.array(results["mean_train_returns"])  # shape (num_seeds, NUM_UPDATES)
    median_train_returns = np.array(results["median_train_returns"])
    eval_returns = np.array(results["eval_returns"])
    best_eval_returns = np.array(results["best_eval_returns"])
    early_stop_flags = np.array(results["early_stop_flags"])

    # Example: log final update's metrics for each seed
    for seed_i in range(num_seeds):
        wandb.log({
            f"final_mean_train_return_seed{seed_i}": float(mean_train_returns[seed_i, -1]),
            f"final_median_train_return_seed{seed_i}": float(median_train_returns[seed_i, -1]),
            f"final_eval_return_seed{seed_i}": float(eval_returns[seed_i, -1]),
            f"best_eval_return_seed{seed_i}": float(best_eval_returns[seed_i, -1]),
            f"did_early_stop_seed{seed_i}": bool(early_stop_flags[seed_i].max()),
        })

    # Optionally, compute average across seeds for a final summary
    final_mean_train_avg = float(mean_train_returns[:, -1].mean())
    final_median_train_avg = float(median_train_returns[:, -1].mean())
    final_eval_avg = float(eval_returns[:, -1].mean())

    wandb.log({
        "final_mean_train_avg_across_seeds": final_mean_train_avg,
        "final_median_train_avg_across_seeds": final_median_train_avg,
        "final_eval_avg_across_seeds": final_eval_avg,
    })

    wandb.finish()
    print("Multi-seed training complete!")