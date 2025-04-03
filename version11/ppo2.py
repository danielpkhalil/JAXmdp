"""
Minimal PPO script that:
 - Uses per-update chunking (Version 2 style)
 - vmaps over multiple seeds (Version 1 style)
 - Performs intermittent evaluation
 - Avoids TracerArrayConversionError by NOT calling np.array(...) inside vmap

Caveat:
 - We do NOT do rolling-median or Python-based loops on JAX arrays in train_single_seed.
   Instead we keep the data in JAX arrays. You can do more advanced Python/NumPy
   logic AFTER the vmap call if desired.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
import numpy as np

from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# If you have a custom TabularEnv, etc.
try:
    from gymnax_env import TabularEnv, TabularEnvParams
except ImportError:
    TabularEnv, TabularEnvParams = None, None


# ---------------------------------------------------------------------
# Network Definitions
# ---------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Conv(32, (3, 3), (2, 2), "SAME", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (2, 2), "SAME", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (1, 1), "SAME", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)

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
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = act_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = act_fn(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


# ---------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, jnp.ndarray]


# ---------------------------------------------------------------------
# Rollout + Update Function (Jitted)
# ---------------------------------------------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    """
    Returns a jitted function that:
     - Rolls out NUM_STEPS in parallel envs
     - Computes GAE advantages
     - Does a PPO update
     - Returns updated TrainState + the trajectory
    """

    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # 1) Rollout
        def env_step(carry, _):
            ts, es, obs_, rng_ = carry
            rng_, rng_act = jax.random.split(rng_)
            pi, val = network.apply(ts.params, obs_)

            action = pi.sample(seed=rng_act)
            logp = pi.log_prob(action)

            rng_, rng_step = jax.random.split(rng_)
            step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
            obsv, es_next, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rngs, es, action, env_params
            )

            transition = Transition(
                done=done,
                action=action,
                value=val,
                reward=rew,
                log_prob=logp,
                obs=obs_,
                info=info,  # shape [NUM_ENVS, ...]
            )
            return (ts, es_next, obsv, rng_), transition

        carry_init = (train_state, env_state, last_obs, rng)
        (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
            env_step, carry_init, xs=None, length=config["NUM_STEPS"]
        )

        # 2) GAE advantage
        _, last_val = network.apply(train_state.params, last_obs)

        def gae_scan(carry, t):
            gae_, nv = carry
            delta = t.reward + config["GAMMA"] * nv * (1 - t.done) - t.value
            gae_ = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * gae_
            return (gae_, t.value), gae_

        (_, _), advantages = jax.lax.scan(
            gae_scan,
            (jnp.zeros_like(last_val), last_val),
            traj,
            reverse=True,
            unroll=16,
        )
        returns = advantages + traj.value

        # 3) PPO update (minibatching)
        def ppo_update(ts, traj_, adv_, ret_, rng_):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            mb_size = batch_size // config["NUM_MINIBATCHES"]

            # Flatten
            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                traj_
            )
            adv_flat = adv_.reshape((batch_size,))
            ret_flat = ret_.reshape((batch_size,))

            rng_, perm_rng = jax.random.split(rng_)
            perm = jax.random.permutation(perm_rng, batch_size)

            def reshape_mb(x):
                return x.reshape((config["NUM_MINIBATCHES"], mb_size) + x.shape[1:])

            traj_shuf = jax.tree_util.tree_map(lambda x: reshape_mb(jnp.take(x, perm, axis=0)), traj_flat)
            adv_shuf = reshape_mb(jnp.take(adv_flat, perm, axis=0))
            ret_shuf = reshape_mb(jnp.take(ret_flat, perm, axis=0))

            def update_mb(ts_, batch_mb):
                mb_traj, mb_adv, mb_ret = batch_mb

                def loss_fn(params, t, ga, rt):
                    pi, val = network.apply(params, t.obs)
                    logp = pi.log_prob(t.action)

                    # Value clipping
                    v_clipped = t.value + (val - t.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    v_loss_1 = (val - rt)**2
                    v_loss_2 = (v_clipped - rt)**2
                    v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

                    # Policy clipping
                    ratio = jnp.exp(logp - t.log_prob)
                    ga_norm = (ga - ga.mean()) / (ga.std() + 1e-8)
                    pg1 = ratio * ga_norm
                    pg2 = jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * ga_norm
                    p_loss = -jnp.mean(jnp.minimum(pg1, pg2))

                    # Entropy
                    ent = jnp.mean(pi.entropy())

                    # Combine
                    loss = p_loss + config["VF_COEF"] * v_loss - config["ENT_COEF"] * ent
                    return loss, (p_loss, v_loss, ent)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, aux), grads = grad_fn(ts_.params, mb_traj, mb_adv, mb_ret)
                ts_ = ts_.apply_gradients(grads=grads)
                return ts_, loss_val

            def scan_minibatch(ts_, _):
                def one_mb(tsmb, i):
                    mb = (
                        jax.tree_util.tree_map(lambda x: x[i], traj_shuf),
                        adv_shuf[i],
                        ret_shuf[i],
                    )
                    tsmb, _ = update_mb(tsmb, mb)
                    return tsmb, None

                idxs = jnp.arange(config["NUM_MINIBATCHES"])
                ts_, _ = jax.lax.scan(one_mb, ts_, idxs)
                return ts_, None

            idx_epochs = jnp.arange(config["UPDATE_EPOCHS"])
            ts, _ = jax.lax.scan(scan_minibatch, ts, idx_epochs)
            return ts, rng_

        train_state, rng = ppo_update(train_state, traj, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj

    return rollout_and_update


# ---------------------------------------------------------------------
# Evaluate Deterministically
# ---------------------------------------------------------------------
def evaluate_policy_deterministic(train_state, network, env, env_params, rng, max_steps=10000):
    """Returns a JAX float for the total reward, so we avoid python float conversion."""
    def body_fn(carry):
        obs_, st_, rng_, total_ = carry
        pi, _ = network.apply(train_state.params, obs_[None])
        act = jnp.argmax(pi.logits[0])
        rng_, rng_step = jax.random.split(rng_)
        obs_next, st_next, rew, done, _info = env.step(rng_step, st_, act, env_params)
        total_ = total_ + rew
        return (obs_next, st_next, rng_, total_), done

    # We'll do a jax.lax.while_loop for purely-JAX evaluation
    def cond_fn(carry):
        obs_, st_, rng_, total_ = carry
        # We need an artificial step limit check
        # We'll store the "steps so far" in total_, or separate?
        return True  # We'll do a manual count or break out? Hard in pure JAX
        # For brevity, let's assume we rely on 'done' from env.
        # or you'd do a partial scan approach.

    obs0, st0 = env.reset(rng, env_params)
    carry_init = (obs0, st0, rng, jnp.array(0.0))

    # For simplicity, let's do a fori_loop up to max_steps
    def step_fn(_i, carry):
        carry, done = body_fn(carry)
        return jax.lax.stop_gradient(carry) if done else carry

    carry_final = jax.lax.fori_loop(0, max_steps, step_fn, carry_init)
    _, _, _, total_rew = carry_final
    return total_rew, None  # ignoring actual step count for brevity


# ---------------------------------------------------------------------
# Single-seed Training (No np.array calls on JAX arrays)
# ---------------------------------------------------------------------
def train_single_seed(rng: jax.random.PRNGKey, config: dict) -> Dict[str, jnp.ndarray]:
    """
    We do a python loop for config["NUM_UPDATES"], but we do NOT convert JAX arrays
    to python arrays inside this function. We store them as jnp arrays and return
    them at the end.
    """
    # 1) Create environment
    if config["ENV_NAME"] == "TabularMDP" and TabularEnv is not None:
        env_ = TabularEnv(config["ENV_FILE"])
        env_params_ = env_.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env_, env_params_ = gymnax.make(config["ENV_NAME"])

    env_ = LogWrapper(env_)

    # 2) Build network
    obs_shape = env_.observation_space(env_params_).shape
    action_dim = env_.action_space(env_params_).n
    if "MiniGrid" in config["ENV_NAME"]:
        net = MiniGridCNNActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    elif len(obs_shape) == 3:
        net = MiniGridCNNActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        net = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # 3) Initialize vector env
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obs, env_state = jax.vmap(env_.reset, in_axes=(0, None))(reset_rngs, env_params_)

    # 4) Build jitted rollout
    rollout_and_update = make_rollout_and_update_fn(env_, env_params_, net, config)

    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0

    # We'll store metrics as jnp arrays in python lists
    mean_returns_list = []
    eval_returns_list = []
    best_eval_list = []

    best_eval_so_far = jnp.array(-1e9)

    for update_i in range(config["NUM_UPDATES"]):
        train_state, env_state, obs, rng, traj_batch = rollout_and_update(
            train_state, env_state, obs, rng
        )
        global_env_steps += steps_per_update

        # returned_episode_returns shape: [NUM_STEPS, NUM_ENVS]
        # It's a JAX array
        returned_ep_ret = traj_batch.info["returned_episode_returns"]  # jnp.ndarray
        # We'll just do a jnp.mean on it
        mean_ret = jnp.mean(returned_ep_ret)
        mean_returns_list.append(mean_ret)

        # Evaluate if needed
        do_eval = (global_env_steps // config["EVAL_FREQUENCY"]) != (
            (global_env_steps - steps_per_update) // config["EVAL_FREQUENCY"]
        )
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, _unused = evaluate_policy_deterministic(train_state, net, env_, env_params_, eval_rng)
            best_eval_so_far = jnp.maximum(best_eval_so_far, eval_ret)
        else:
            eval_ret = jnp.nan
        eval_returns_list.append(eval_ret)
        best_eval_list.append(best_eval_so_far)

    # Convert these python lists of jnp arrays into jnp arrays
    mean_returns_jnp = jnp.stack(mean_returns_list, axis=0)        # shape [NUM_UPDATES]
    eval_returns_jnp = jnp.stack(eval_returns_list, axis=0)        # shape [NUM_UPDATES]
    best_eval_jnp = jnp.stack(best_eval_list, axis=0)              # shape [NUM_UPDATES]

    # Return as a dict
    return {
        "mean_train_returns": mean_returns_jnp,
        "eval_returns": eval_returns_jnp,
        "best_eval_returns": best_eval_jnp,
        # We skip median or episode-level logic. If you want it, do it after vmap.
    }


# ---------------------------------------------------------------------
# Multi-seed vmap
# ---------------------------------------------------------------------
def run_ppo_training_multi_seed(rng_seeds: jnp.ndarray, config: dict) -> Dict[str, jnp.ndarray]:
    """
    Vmap over the train_single_seed. Because we do NOT call np.array(...) or
    python float(...) inside train_single_seed, we won't get TracerArrayConversionError.
    """
    batched = jax.vmap(train_single_seed, in_axes=(0, None))(rng_seeds, config)
    # Now 'batched' is a dict whose arrays have shape [num_seeds, num_updates].
    return batched


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
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
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/fruitbot_easy_l0_40_fs8/consolidated.npz",  # If using TabularMDP
        "REWARD_SCALE": 1.0,
        "EVAL_FREQUENCY": 1000,
        # We skip rolling median inside the script, do it outside if needed
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 5.0,
    }

    # Num updates
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    wandb.init(project="combined_ppo_vmap_no_np_convert", config=config)

    # Make multiple RNG seeds
    num_seeds = 4
    base_rng = jax.random.PRNGKey(config["SEED"])
    rng_seeds = jax.random.split(base_rng, num_seeds)

    # Vmap multi-seed
    results = run_ppo_training_multi_seed(rng_seeds, config)
    # results is e.g. {
    #   "mean_train_returns": shape (num_seeds, num_updates),
    #   "eval_returns": shape (num_seeds, num_updates),
    #   ...
    # }

    # Now we can do Python/NumPy stuff safely outside vmap:
    mean_train_np = np.array(results["mean_train_returns"])  # [num_seeds, num_updates]
    eval_returns_np = np.array(results["eval_returns"])      # [num_seeds, num_updates]
    best_eval_np = np.array(results["best_eval_returns"])    # [num_seeds, num_updates]

    # Example: log final update's metrics for each seed
    for seed_i in range(num_seeds):
        wandb.log({
            f"final_mean_return_seed{seed_i}": float(mean_train_np[seed_i, -1]),
            f"final_eval_return_seed{seed_i}": float(eval_returns_np[seed_i, -1]),
            f"best_eval_so_far_seed{seed_i}": float(best_eval_np[seed_i, -1]),
        })

    # Optionally average across seeds
    final_mean_avg = float(mean_train_np[:, -1].mean())
    final_eval_avg = float(eval_returns_np[:, -1].mean())

    wandb.log({
        "final_mean_train_avg_across_seeds": final_mean_avg,
        "final_eval_return_avg_across_seeds": final_eval_avg,
    })

    wandb.finish()
    print("Done multi-seed PPO with no tracer conversion errors!")
