"""
ppo_minigrid_wandb_cnn.py

Example JAX PPO script modified to match the SB3 “MiniGridCNN” architecture
exactly, with minimal CSV logging and optional action-mask support.
"""

import csv
import json
import os
import pprint
import time
from typing import NamedTuple, TypedDict

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment
from gymnax.wrappers.purerl import LogWrapper
from gymnax.environments import spaces  # ← NEW: to inspect Dict obs spaces


# --------------------------------------------------------------------------
# Helper to zero-out invalid actions
# --------------------------------------------------------------------------
def _mask_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Very negative logits for actions whose mask==0."""
    return logits + (1.0 - mask) * -1e9


class PPOArgs(TypedDict):
    lr: float
    num_envs: int
    num_steps: int
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    activation: str


# --------------------------------------------------------------------------
# MiniGridCNN Actor-Critic
# --------------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    """
    Three conv layers + 512-unit FC exactly matching SB3 MiniGridCNN, now with
    optional masking of invalid actions.
    """

    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Accept either raw tensor or {"obs":tensor, "action_mask":mask}
        if isinstance(x, dict):
            mask = x["action_mask"]
            x = x["obs"]
        else:
            mask = jnp.ones((x.shape[0], self.action_dim), dtype=jnp.float32)

        x = x.astype(jnp.float32)
        x = nn.Conv(
            32, (3, 3), (2, 2), "SAME", kernel_init=orthogonal(np.sqrt(2))
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64, (3, 3), (2, 2), "SAME", kernel_init=orthogonal(np.sqrt(2))
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64, (3, 3), (1, 1), "SAME", kernel_init=orthogonal(np.sqrt(2))
        )(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        logits = _mask_logits(logits, mask)  # ← NEW
        pi = distrax.Categorical(logits=logits)

        value = nn.Dense(1, kernel_init=orthogonal(0.01))(x)
        return pi, jnp.squeeze(value, axis=-1)


# --------------------------------------------------------------------------
# Two-layer MLP Actor-Critic
# --------------------------------------------------------------------------
class MLPActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if isinstance(x, dict):
            mask = x["action_mask"]
            x = x["obs"]
        else:
            mask = jnp.ones((x.shape[0], self.action_dim), dtype=jnp.float32)

        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = act_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = act_fn(x)

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        logits = _mask_logits(logits, mask)  # ← NEW
        pi = distrax.Categorical(logits=logits)

        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


# --------------------------------------------------------------------------
# Transition container
# --------------------------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: object  # may be Dict or ndarray
    info: jnp.ndarray


def explained_var(y, pred):
    y_var = jnp.var(y)
    pred_var = jnp.var(pred)
    return jnp.maximum(-1.0, 1 - pred_var / (y_var + 1e-8))


# --------------------------------------------------------------------------
# Rollout collection + PPO update
# --------------------------------------------------------------------------
def make_rollout_and_update_fn(
    env: Environment, env_params, network, algo_args: PPOArgs
):
    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # 1) Collect rollout -------------------------------------------------
        def env_step_fn(carry, _):
            train_state, env_state, last_obs, rng = carry
            rng, act_rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, last_obs)
            action = pi.sample(seed=act_rng)
            log_prob = pi.log_prob(action)

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, algo_args["num_envs"])
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
            env_step_fn, carry_init, None, length=algo_args["num_steps"]
        )
        train_state, env_state, last_obs, rng = carry_out

        # 2) Generalised advantage estimation ------------------------------
        _, last_val = network.apply(train_state.params, last_obs)

        def gae_scan_fn(carry, transition):
            gae, next_val = carry
            delta = (
                transition.reward
                + algo_args["gamma"] * next_val * (1.0 - transition.done)
                - transition.value
            )
            gae = (
                delta
                + algo_args["gamma"]
                * algo_args["gae_lambda"]
                * (1.0 - transition.done)
                * gae
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

        # 3) PPO update ------------------------------------------------------
        def ppo_update(train_state, traj_batch, advantages, returns, rng):
            batch_size = algo_args["num_steps"] * algo_args["num_envs"]
            minibatch_size = algo_args["minibatch_size"]
            num_minibatches = batch_size // minibatch_size

            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
            )
            adv_flat = advantages.reshape((batch_size,))
            ret_flat = returns.reshape((batch_size,))

            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)

            def reshape_mb(x):
                return x.reshape((num_minibatches, minibatch_size) + x.shape[1:])

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

                    v_clipped = t.value + (value - t.value).clip(
                        -algo_args["clip_eps"], algo_args["clip_eps"]
                    )
                    v_loss_1 = (value - rt) ** 2
                    v_loss_2 = (v_clipped - rt) ** 2
                    value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

                    ratio = jnp.exp(log_prob - t.log_prob)
                    ga_normed = (ga - ga.mean()) / (ga.std() + 1e-8)
                    pg_loss_1 = ratio * ga_normed
                    pg_loss_2 = (
                        jnp.clip(
                            ratio,
                            1.0 - algo_args["clip_eps"],
                            1.0 + algo_args["clip_eps"],
                        )
                        * ga_normed
                    )
                    policy_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                    entropy = jnp.mean(pi.entropy())
                    vf_explained_var = explained_var(rt, value)

                    metrics = {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": entropy,
                        "vf_explained_var": vf_explained_var,
                    }

                    loss = (
                        policy_loss
                        + algo_args["vf_coef"] * value_loss
                        - algo_args["ent_coef"] * entropy
                    )
                    return loss, metrics

                loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, metrics), grads = loss_grad_fn(
                    train_state.params, traj_mb, adv_mb, ret_mb
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss_val, metrics

            def scan_minibatch_fn(train_state, i):
                traj_mb = jax.tree_util.tree_map(lambda x: x[i], traj_shuf)
                adv_mb = adv_shuf[i]
                ret_mb = ret_shuf[i]
                train_state, _, metrics = update_minibatch(
                    train_state, (traj_mb, adv_mb, ret_mb)
                )
                return train_state, metrics

            for _ in range(algo_args["update_epochs"]):
                indices = jnp.arange(num_minibatches)
                train_state, metrics_stack = jax.lax.scan(
                    scan_minibatch_fn, train_state, indices
                )
                metrics_mean = jax.tree_util.tree_map(
                    lambda x: jnp.mean(jnp.stack(x), axis=0), metrics_stack
                )

            return train_state, rng, metrics_mean

        train_state, rng, metrics = ppo_update(
            train_state, traj_batch, advantages, returns, rng
        )
        return train_state, env_state, last_obs, rng, traj_batch, metrics

    return rollout_and_update


# --------------------------------------------------------------------------
# Deterministic evaluation
# --------------------------------------------------------------------------
def evaluate_policy_deterministic(
    train_state, network, env, env_params, rng, max_steps=10000
):
    obs, state = env.reset(rng, env_params)
    done = False
    total_reward = 0.0
    steps = 0
    while (not done) and (steps < max_steps):
        net_input = (
            {"obs": obs["obs"][None, ...], "action_mask": obs["action_mask"][None, ...]}
            if isinstance(obs, dict)
            else obs[None, ...]
        )
        pi, _ = network.apply(train_state.params, net_input)
        action = int(jnp.argmax(pi.logits[0]))
        obs, state, reward, done, info = env.step(rng, state, action, env_params)
        total_reward += float(reward)
        steps += 1
    return total_reward, steps


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------
def run_ppo_training(
    env: Environment,
    env_params: dict,
    total_timesteps: int,
    seed: int,
    eval_frequency: int,
    stop_on_eval_reward: float,
    stop_on_median_train_reward: float,
    algo_args: PPOArgs,
    log_dir: str,
):
    # --- CSV logging -------------------------------------------------------
    csv_file = open(os.path.join(log_dir, "metrics.csv"), "w", newline="")
    fieldnames = [
        "training_step",
        "fps",
        "num_episodes",
        "global_env_steps",
        "median_train_return",
        "mean_train_return",
        "eval_return",
        "best_eval_return",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/vf_explained_var",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    env = LogWrapper(env)

    # Determine observation type (plain Box or Dict with mask)
    obs_space = env.observation_space(env_params)
    action_dim = env.action_space(env_params).n

    if isinstance(obs_space, spaces.Dict):
        dummy_obs = {
            "obs": jnp.zeros(
                (1,) + obs_space.spaces["obs"].shape, dtype=obs_space.spaces["obs"].dtype
            ),
            "action_mask": jnp.ones((1, action_dim), dtype=jnp.float32),
        }
        uses_cnn = len(obs_space.spaces["obs"].shape) == 3
    else:
        dummy_obs = jnp.zeros((1,) + obs_space.shape, dtype=obs_space.dtype)
        uses_cnn = len(obs_space.shape) == 3

    if uses_cnn:
        network = MiniGridCNNActorCritic(action_dim=action_dim)
    else:
        network = MLPActorCritic(
            action_dim=action_dim, activation=algo_args["activation"]
        )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(algo_args["max_grad_norm"]),
        optax.adam(algo_args["lr"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, algo_args["num_envs"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    rollout_and_update_fn = make_rollout_and_update_fn(
        env, env_params, network, algo_args
    )

    steps_per_update = algo_args["num_envs"] * algo_args["num_steps"]
    global_env_steps = 0

    train_returns_buffer = []
    best_eval_return = -np.inf

    num_updates = int(total_timesteps // steps_per_update)
    for training_step in range(num_updates):
        step_start = time.time()
        (
            train_state,
            env_state,
            obsv,
            rng,
            traj_batch,
            train_metrics,
        ) = rollout_and_update_fn(train_state, env_state, obsv, rng)
        global_env_steps += steps_per_update

        info_dict = traj_batch.info
        returned_ep_ret = np.array(info_dict["returned_episode_returns"])
        returned_ep = np.array(info_dict["returned_episode"])
        ended_idx = np.where(returned_ep > 0)
        ep_returns = returned_ep_ret[ended_idx]
        for r in ep_returns:
            train_returns_buffer.append(r)

        # Run deterministic evaluation on schedule
        do_eval = (global_env_steps // eval_frequency) != (
            (global_env_steps - steps_per_update) // eval_frequency
        )
        eval_ret = None
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, eval_steps = evaluate_policy_deterministic(
                train_state, network, env, env_params, eval_rng
            )
            best_eval_return = max(best_eval_return, eval_ret)

        recent_returns = train_returns_buffer[-100:]
        median_train_return = (
            float(np.median(recent_returns)) if recent_returns else 0.0
        )
        mean_train_return = float(np.mean(recent_returns)) if recent_returns else 0.0

        fps = steps_per_update / (time.time() - step_start)

        # Log to CSV
        metrics = {
            "training_step": training_step,
            "fps": fps,
            "num_episodes": len(train_returns_buffer),
            "global_env_steps": global_env_steps,
            "median_train_return": median_train_return,
            "mean_train_return": mean_train_return,
            "eval_return": eval_ret if eval_ret is not None else float("nan"),
            "best_eval_return": best_eval_return,
        }
        for metric_key, metric_value in train_metrics.items():
            metrics[f"train/{metric_key}"] = float(metric_value)

        csv_writer.writerow(metrics)
        pprint.pp(metrics)
        print()

        # Early stopping
        if eval_ret is not None and eval_ret >= stop_on_eval_reward:
            print(
                f"Stopping training after {global_env_steps} env steps "
                f"(eval reward: {eval_ret:.3f})."
            )
            break
        elif median_train_return >= stop_on_median_train_reward:
            print(
                f"Stopping training after {global_env_steps} env steps "
                f"(median train reward: {median_train_return:.3f})."
            )
            break

    print("Training finished.")
    csv_file.close()
    return train_state
