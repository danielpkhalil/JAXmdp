"""
Combined PPO script that:
 - Uses per-update chunking (hybrid version)
 - vmaps over multiple seeds
 - Tracks mean return per update for each seed
 - Plots each seed's learning curve and logs them to wandb

Caveats:
 - Because of vmap, early stopping per seed is not possible.
 - All plotting and logging is done after training, so that it doesn't interfere with vmap.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
import numpy as np
import matplotlib.pyplot as plt

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


# -----------------------------------------------------------------------------
# Network Definitions
# -----------------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Conv(32, (3,3), (2,2), "SAME", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3,3), (2,2), "SAME", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3,3), (1,1), "SAME", kernel_init=orthogonal(jnp.sqrt(2)))(x)
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
        act_fn = nn.relu if self.activation=="relu" else nn.tanh
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = act_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
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
    info: Dict[str, jnp.ndarray]


# -----------------------------------------------------------------------------
# Rollout + PPO Update Function (jitted)
# -----------------------------------------------------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # Rollout step: collect NUM_STEPS of transitions
        def env_step(carry, _):
            ts, es, obs_, rng_ = carry
            rng_, rng_act = jax.random.split(rng_)
            pi, val = network.apply(ts.params, obs_)
            action = pi.sample(seed=rng_act)
            logp = pi.log_prob(action)
            rng_, rng_step = jax.random.split(rng_)
            step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
            obsv, es_next, rew, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                step_rngs, es, action, env_params)
            transition = Transition(
                done=done, action=action, value=val,
                reward=rew, log_prob=logp, obs=obs_, info=info
            )
            return (ts, es_next, obsv, rng_), transition

        carry_init = (train_state, env_state, last_obs, rng)
        (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
            env_step, carry_init, None, length=config["NUM_STEPS"]
        )
        # Compute advantage using GAE
        _, last_val = network.apply(train_state.params, last_obs)
        def gae_scan(carry, t):
            gae, nv = carry
            delta = t.reward + config["GAMMA"] * nv * (1 - t.done) - t.value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * gae
            return (gae, t.value), gae
        (_, _), advantages = jax.lax.scan(
            gae_scan, (jnp.zeros_like(last_val), last_val), traj,
            reverse=True, unroll=16
        )
        returns = advantages + traj.value

        # PPO update with minibatching
        def ppo_update(ts, traj_, adv_, ret_, rng_):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            mb_size = batch_size // config["NUM_MINIBATCHES"]
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
            def update_mb(ts_, mb):
                mb_traj, mb_adv, mb_ret = mb
                def loss_fn(params, t, ga, rt):
                    pi, val = network.apply(params, t.obs)
                    logp = pi.log_prob(t.action)
                    v_clipped = t.value + (val - t.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    v_loss = 0.5 * jnp.mean(jnp.maximum((val - rt)**2, (v_clipped - rt)**2))
                    ratio = jnp.exp(logp - t.log_prob)
                    ga_norm = (ga - ga.mean())/(ga.std()+1e-8)
                    p_loss = -jnp.mean(jnp.minimum(ratio * ga_norm,
                                                   jnp.clip(ratio, 1-config["CLIP_EPS"], 1+config["CLIP_EPS"]) * ga_norm))
                    ent = jnp.mean(pi.entropy())
                    loss = p_loss + config["VF_COEF"] * v_loss - config["ENT_COEF"] * ent
                    return loss, (p_loss, v_loss, ent)
                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, aux), grads = grad_fn(ts_.params, mb_traj, mb_adv, mb_ret)
                ts_ = ts_.apply_gradients(grads=grads)
                return ts_, loss_val
            def scan_mb(ts_, _):
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
            ts, _ = jax.lax.scan(scan_mb, ts, idx_epochs)
            return ts, rng_
        train_state, rng = ppo_update(train_state, traj, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj
    return rollout_and_update


# -----------------------------------------------------------------------------
# Deterministic Evaluation (using jax.lax.while_loop)
# -----------------------------------------------------------------------------
def evaluate_policy_deterministic(train_state, network, env, env_params, rng, max_steps=10000):
    def cond_fn(carry):
        obs, st, rng, total, steps, done = carry
        return jnp.logical_and(steps < max_steps, jnp.logical_not(done))
    def body_fn(carry):
        obs, st, rng, total, steps, done = carry
        pi, _ = network.apply(train_state.params, obs[None])
        action = jnp.argmax(pi.logits[0])
        rng, rng_step = jax.random.split(rng)
        obs_next, st_next, rew, done, info = env.step(rng_step, st, action, env_params)
        return (obs_next, st_next, rng, total + rew, steps + 1, done)
    obs0, st0 = env.reset(rng, env_params)
    init_carry = (obs0, st0, rng, jnp.array(0.0), jnp.array(0), jnp.array(False))
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return final_carry[3], final_carry[4]


# -----------------------------------------------------------------------------
# Single-Seed Training (functional, no np.array conversions)
# -----------------------------------------------------------------------------
def train_single_seed(rng: jax.random.PRNGKey, config: dict) -> Dict[str, jnp.ndarray]:
    # Setup environment
    if config["ENV_NAME"] == "TabularMDP" and TabularEnv is not None:
        env_, env_params_ = TabularEnv(config["ENV_FILE"]), None
        env_params_ = env_.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        env_, env_params_ = gymnax.make(config["ENV_NAME"])
    env_ = LogWrapper(env_)
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
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obs, env_state = jax.vmap(env_.reset, in_axes=(0, None))(reset_rngs, env_params_)
    rollout_and_update = make_rollout_and_update_fn(env_, env_params_, net, config)
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0
    # We'll store the mean return per update
    mean_returns_list = []
    eval_returns_list = []
    best_eval_list = []
    best_eval_so_far = -1e9
    for update_i in range(config["NUM_UPDATES"]):
        train_state, env_state, obs, rng, traj = rollout_and_update(train_state, env_state, obs, rng)
        global_env_steps += steps_per_update
        # Compute mean return (using pure JAX)
        mean_ret = jnp.mean(traj.info["returned_episode_returns"])
        mean_returns_list.append(mean_ret)
        # Evaluate periodically
        do_eval = (global_env_steps // config["EVAL_FREQUENCY"]) != (
            (global_env_steps - steps_per_update) // config["EVAL_FREQUENCY"]
        )
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            eval_ret, _ = evaluate_policy_deterministic(train_state, net, env_, env_params_, eval_rng)
            best_eval_so_far = jnp.maximum(best_eval_so_far, eval_ret)
        else:
            eval_ret = jnp.nan
        eval_returns_list.append(eval_ret)
        best_eval_list.append(best_eval_so_far)
    mean_returns_jnp = jnp.stack(mean_returns_list, axis=0)       # shape: [NUM_UPDATES]
    eval_returns_jnp = jnp.stack(eval_returns_list, axis=0)           # shape: [NUM_UPDATES]
    best_eval_jnp = jnp.stack(best_eval_list, axis=0)                 # shape: [NUM_UPDATES]
    return {
        "mean_train_returns": mean_returns_jnp,
        "eval_returns": eval_returns_jnp,
        "best_eval_returns": best_eval_jnp,
    }


# -----------------------------------------------------------------------------
# Multi-Seed Training via vmap
# -----------------------------------------------------------------------------
def run_ppo_training_multi_seed(rng_seeds: jnp.ndarray, config: dict) -> Dict[str, jnp.ndarray]:
    return jax.vmap(train_single_seed, in_axes=(0, None))(rng_seeds, config)


# -----------------------------------------------------------------------------
# Main Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
    }
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    wandb.init(project="combined_ppo_vmap_plots", config=config)

    num_seeds = 4
    base_rng = jax.random.PRNGKey(config["SEED"])
    rng_seeds = jax.random.split(base_rng, num_seeds)

    results = run_ppo_training_multi_seed(rng_seeds, config)
    # results is a dict of arrays with shape [num_seeds, NUM_UPDATES]
    mean_train_returns = np.array(results["mean_train_returns"])  # shape: [num_seeds, NUM_UPDATES]
    eval_returns = np.array(results["eval_returns"])
    best_eval_returns = np.array(results["best_eval_returns"])

    # Create a plot that shows the return curve for each seed over update steps.
    plt.figure(figsize=(10,6))
    for seed in range(num_seeds):
        plt.plot(mean_train_returns[seed], label=f"Seed {seed}")
    plt.xlabel("Update Step")
    plt.ylabel("Mean Episode Return")
    plt.title("Training Returns per Update (per Seed)")
    plt.legend()
    wandb.log({"training_returns_plot": wandb.Image(plt)})
    plt.close()

    # Optionally, create a plot for evaluation returns.
    plt.figure(figsize=(10,6))
    for seed in range(num_seeds):
        plt.plot(eval_returns[seed], label=f"Seed {seed}")
    plt.xlabel("Update Step")
    plt.ylabel("Eval Return")
    plt.title("Evaluation Returns per Update (per Seed)")
    plt.legend()
    wandb.log({"eval_returns_plot": wandb.Image(plt)})
    plt.close()

    # Log final numeric metrics (averaged over seeds)
    final_mean = float(mean_train_returns[:, -1].mean())
    final_eval = float(eval_returns[:, -1].mean())
    wandb.log({
        "final_mean_train_return_avg": final_mean,
        "final_eval_return_avg": final_eval,
    })

    wandb.finish()
    print("Multi-seed training complete and plots logged to wandb!")
