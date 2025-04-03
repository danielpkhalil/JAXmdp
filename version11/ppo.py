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

# If you have your custom TabularEnv definitions:
try:
    from gymnax_env import TabularEnv, TabularEnvParams
except ImportError:
    TabularEnv, TabularEnvParams = None, None


# -----------------------------------------------------------------------------
# Actor-Critic Networks
# -----------------------------------------------------------------------------
class MiniGridCNNActorCritic(nn.Module):
    """
    CNN architecture replicating SB3's MiniGridCNN (3xConv => Flatten => 512 => policy/value).
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x is (batch, H, W, C), cast to float32
        x = x.astype(jnp.float32)
        # Conv layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Policy head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        # Value head
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


class MLPActorCritic(nn.Module):
    """
    Simple 2x64 MLP for non-image observations
    """
    action_dim: int
    activation: str = "tanh"  # or "relu"

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
      - Initializes the environment and the policy network
      - Runs a fully-jitted PPO loop for `NUM_UPDATES`
      - Returns a dict with final runner state + collected metrics

    NOTE: Some config keys (EVAL_FREQUENCY, TRAIN_MEDIAN_WINDOW, OPTIMAL_REWARD, etc.)
    are NOT used inside the fully-jitted loop. If you need on-the-fly eval or
    early-stopping, do it outside the jit scanning.
    """

    # Calculate how many updates from config:
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    # Create environment
    if config["ENV_NAME"] == "TabularMDP" and TabularEnv is not None:
        env = TabularEnv(config["ENV_FILE"])
        # Possibly override default params
        env_params = env.default_params().replace(
            reward_scale=config.get("REWARD_SCALE", 1.0)
        )
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])

    # Wrap environment
    env = LogWrapper(env)

    # Learning rate schedule if needed:
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # The train(rng) function we will JIT
    def train(rng):
        # Decide on CNN vs MLP
        obs_shape = env.observation_space(env_params).shape
        action_dim = env.action_space(env_params).n

        if "MiniGrid" in config["ENV_NAME"]:
            network = MiniGridCNNActorCritic(action_dim=action_dim)
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
        elif len(obs_shape) == 3:
            # Some other image-based environment
            network = MiniGridCNNActorCritic(action_dim=action_dim)
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
        else:
            # MLP fallback
            network = MLPActorCritic(action_dim=action_dim, activation=config.get("ACTIVATION", "relu"))
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

        # Initialize network params
        rng, init_rng = jax.random.split(rng)
        network_params = network.init(init_rng, dummy_obs)

        # Optimizer
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

        # Reset env (vector of envs)
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # ---------------------------------------------------------------------
        # One update step (rollout + PPO)
        # ---------------------------------------------------------------------
        def _update_step(runner_state, update_idx):
            """
            Collect NUM_STEPS of data, then run PPO update.
            Returns new (train_state, env_state, last_obs, rng, global_step_count).
            """
            train_state, env_state, last_obs, rng, global_step_count = runner_state

            # (A) Rollout
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

            # (B) GAE advantage
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

            # (C) PPO update
            def _update_epoch(update_state, _):
                def _update_minibatch(ts, minibatch):
                    mb_traj, mb_adv, mb_ret = minibatch

                    def loss_fn(params, t, adv_, rt_):
                        pi, val = network.apply(params, t.obs)
                        logp = pi.log_prob(t.action)

                        # Value clipping
                        v_clipped = t.value + (val - t.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        v_loss_1 = (val - rt_) ** 2
                        v_loss_2 = (v_clipped - rt_) ** 2
                        value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

                        # Policy clipping
                        ratio = jnp.exp(logp - t.log_prob)
                        adv_norm = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
                        pg_loss_1 = ratio * adv_norm
                        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                                             1.0 + config["CLIP_EPS"]) * adv_norm
                        policy_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        # Combine
                        loss = (policy_loss
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy)
                        return loss, (policy_loss, value_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss_val, _aux), grads = grad_fn(ts.params, mb_traj, mb_adv, mb_ret)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, loss_val

                ts, traj_b, adv_b, ret_b, k_ = update_state

                # Shuffle entire batch
                k_, pk = jax.random.split(k_)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                perm = jax.random.permutation(pk, batch_size)

                # Flatten [T, N, ...] => [batch_size, ...]
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

                # Repeat for UPDATE_EPOCHS
                for _ in range(config["UPDATE_EPOCHS"]):
                    indices = jnp.arange(config["NUM_MINIBATCHES"])
                    ts, _ = jax.lax.scan(scan_minibatch, ts, indices)

                new_state = (ts, traj_b, adv_b, ret_b, k_)
                return new_state, None

            update_state = (train_state, traj_batch, advantages, returns, rng)
            # We do exactly 1 call to _update_epoch—but if you want multiple epochs,
            # note that we do them in the 'for _ in range(UPDATE_EPOCHS)' above.
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, length=1)
            train_state = update_state[0]
            rng = update_state[-1]

            # Store metrics from the rollout
            metrics = traj_batch.info

            # Increase global step count
            global_step_count += (config["NUM_ENVS"] * config["NUM_STEPS"])

            new_runner_state = (train_state, env_state, last_obs, rng, global_step_count)
            return new_runner_state, metrics

        # ---------------------------------------------------------------------
        # Scan over all updates
        # ---------------------------------------------------------------------
        runner_state = (train_state, env_state, obsv, rng, jnp.array(0))
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        return {
            "runner_state": runner_state,
            "metrics": metrics  # shape => [NUM_UPDATES, NUM_STEPS, NUM_ENVS, ...]
        }

    return train


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e4,
        "UPDATE_EPOCHS": 4,   # Matches SB3's typical PPO n_epochs
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

        # The following can't be done inside the fully-jitted loop:
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 5.0,
        "ANNEAL_LR": True,   # Use linear LR schedule
    }

    # Build the train(rng) function
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    # Single-seed example
    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    print(f"Single seed training took {time.time() - t0:.2f} seconds.")

    # The shape of returned_episode_returns is [NUM_UPDATES, NUM_STEPS, NUM_ENVS]
    returned_ep_ret = out["metrics"]["returned_episode_returns"]
    # For a simple scalar curve, we might average over steps & envs each update:
    mean_return_per_update = returned_ep_ret.mean(axis=(-1, -2))

    plt.plot(mean_return_per_update)
    plt.xlabel("Update")
    plt.ylabel("Mean Episode Return")
    plt.title("Single-Seed PPO on TabularMDP (fully-jitted)")
    plt.show()

    # Multi-seed example:
    num_seeds = 4
    rngs = jax.random.split(jax.random.PRNGKey(999), num_seeds)
    batched_train = jax.jit(jax.vmap(train_fn))
    t0 = time.time()
    outs = jax.block_until_ready(batched_train(rngs))
    print(f"{num_seeds}-seed training took {time.time() - t0:.2f} seconds.")

    # Each seed has shape [NUM_UPDATES, NUM_STEPS, NUM_ENVS]
    rets_all = outs["metrics"]["returned_episode_returns"]  # shape => (num_seeds, updates, steps, envs)
    for i in range(num_seeds):
        mean_ret_i = rets_all[i].mean(axis=(-1, -2))
        plt.plot(mean_ret_i, label=f"seed {i}")

    plt.xlabel("Update")
    plt.ylabel("Mean Episode Return")
    plt.title("Multi-Seed PPO on TabularMDP")
    plt.legend()
    plt.show()
