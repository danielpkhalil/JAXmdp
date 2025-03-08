"""
ppo_tabular_wandb_cnn_universal_debug_batchdim.py

Key differences:
- We always init the network with shape (1, H, W, C).
- We always pass (N, H, W, C) to the CNN at train time.
- Includes debug prints to see shapes.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb   # for logging

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Any

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# If you have a custom env:
from gymnax_env import TabularEnv, TabularEnvParams

import matplotlib.pyplot as plt

# ------------------------------
# CNN-based Actor-Critic
# ------------------------------
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        """
        Expects x shape: (batch, H, W, C).
        Example steps: 2 Convs (stride=2), global avg pool, 1 Dense, policy & value heads.
        """
        # Conv #1
        x = nn.Conv(
            features=32, kernel_size=(3,3), strides=(2,2),
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        # Conv #2
        x = nn.Conv(
            features=64, kernel_size=(3,3), strides=(2,2),
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        # Global average pooling: reduce (H, W) => just keep channels
        x = x.mean(axis=(1, 2))  # shape: (batch, 64)

        # Dense layer
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # Policy head
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        pi = distrax.Categorical(logits=logits)

        # Value head
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        value = jnp.squeeze(critic, axis=-1)

        return pi, value


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
# Training
# ------------------------------
def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # 1) Create env
    if config["ENV_NAME"] == "TabularMDP":
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params()
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)

    # 2) LR schedule
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # 3) train function
    def train(rng):
        network = ActorCritic(action_dim=env.action_space(env_params).n)
        rng, init_rng = jax.random.split(rng)

        # We ALWAYS init with shape (1, H, W, C) to include a batch dimension.
        obs_shape = env.observation_space(env_params).shape  # (H, W, C)
        if config["DEBUG"]:
            print("DEBUG: env.observation_space shape =", obs_shape)

        init_obs = jnp.zeros((1,) + obs_shape)  # batch=1
        if config["DEBUG"]:
            print("DEBUG: init_obs shape for init =", init_obs.shape)

        network_params = network.init(init_rng, init_obs)

        # optimizer
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
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Env init
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
        # obsv shape => (NUM_ENVS, H, W, C)

        # Optional debug pass
        if config["DEBUG"]:
            print("DEBUG: real initial obsv shape =", obsv.shape)
            # single batch
            single_obs = obsv[:1]  # shape (1, H, W, C)
            print("DEBUG: single_obs shape for debug =", single_obs.shape)

            pi_debug, val_debug = network.apply(train_state.params, single_obs)
            print("DEBUG: pi_debug.logits shape =", pi_debug.logits.shape,
                  " val_debug.shape =", val_debug.shape)

        # -----------
        # SCAN
        # -----------
        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state

            # 1) gather rollout
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                rng, act_rng = jax.random.split(rng)

                # pass entire batch (N, H, W, C)
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
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng),
                None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state

            # 2) GAE advantage
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1.0 - transition.done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    )
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # 3) PPO Update
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.mean(
                            jnp.maximum(value_losses, value_losses_clipped)
                        )

                        # Policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_normed = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_normed
                        loss_actor2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * gae_normed
                        loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state

                # Shuffle entire trajectory
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)

                # Flatten from [T, N_env, ...] to [T*N_env, ...]
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    batch,
                )
                # Shuffle
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch,
                )
                # Reshape -> minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled,
                )

                # Scan minibatches
                train_state, losses = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metric = traj_batch.info

            # debug callback
            if config.get("DEBUG"):
                def debug_callback(info):
                    rets = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={rets[t]}")
                jax.debug.callback(debug_callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        # run entire training
        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {
            "runner_state": runner_state,
            "metrics": metrics,
        }

    return train


# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e3,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": True,
        "DEBUG": True,  # turn on shape debugging
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0/consolidated.npz",
    }

    wandb.init(project="my_tabular_ppo_cnn_batchdim_debug", config=config)
    rng = jax.random.PRNGKey(42)

    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    out = train_jit(rng)
    print("Training finished. Final output keys:", out.keys())

    metrics = jax.tree_util.tree_map(np.array, out["metrics"])
    if "returned_episode_returns" in metrics:
        try:
            mean_returns = metrics["returned_episode_returns"].mean(axis=-1)
        except:
            mean_returns = metrics["returned_episode_returns"]
        mean_returns = mean_returns.reshape(-1)
        for i, ret in enumerate(mean_returns):
            wandb.log({"update_step": i, "mean_return": ret})

        plt.figure()
        plt.plot(mean_returns, label="Mean Return")
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.title("PPO Training Performance")
        plt.legend()
        plt.tight_layout()
        wandb.log({"training_returns_plot": wandb.Image(plt)})
        plt.show()

    wandb.finish()
