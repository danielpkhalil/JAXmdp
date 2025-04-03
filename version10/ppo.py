"""
ppo_atari_framestack.py

Example JAX PPO script with the same structure as ppo_minigrid_wandb_cnn.py,
but adapted for Atari-like CNN and optional framestacking.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb

from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple, Any, Tuple, Dict, Union

import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper

# ------------------------------------------------------
# 1) Example: AtariCNNActorCritic with orthogonal init
# ------------------------------------------------------
class AtariCNNActorCritic(nn.Module):
    """
    A standard "Nature CNN"-style architecture for Atari:
      1) Conv2D(32, kernel=8, stride=4)
      2) Conv2D(64, kernel=4, stride=2)
      3) Conv2D(64, kernel=3, stride=1)
      4) Flatten
      5) Dense(512) + ReLU
      => Outputs policy (Categorical) logits & value.
    """
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, H, W, C * framestack)
        # Typically x is uint8 [0..255]. Convert to float32, optionally /255
        x = x.astype(jnp.float32) / 255.0

        # Convolution layers
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # FC layer (512)
        x = nn.Dense(features=512, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Policy head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        # Value head
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return pi, jnp.squeeze(value, axis=-1)


# ------------------------------------------------------
# 2) MLP Actor-Critic (fallback for non-image envs)
# ------------------------------------------------------
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


# ------------------------------------------------------
# 3) Optional Framestack Wrapper for Atari
# ------------------------------------------------------
# If you want to natively handle stacked frames with Gymnax,
# you can create a simple wrapper that stores the last K frames.
# This is a minimal example; for a more robust approach,
# see "atari_wrappers.py" or RL libraries like SB3.
class FrameStackEnv:
    def __init__(self, env, num_stack: int):
        self.env = env
        self.num_stack = num_stack

        # Original observation space
        ob_space = env.observation_space(env.default_params())
        # e.g., shape = (H, W, C). We'll expand last dim to C * num_stack
        self.obs_shape = ob_space.shape
        self.stacked_shape = (self.obs_shape[0], self.obs_shape[1],
                              self.obs_shape[2] * self.num_stack)

    def observation_space(self, params):
        # Return a Box with shape = (H, W, C * framestack)
        from gymnax.environments import spaces
        low_val = 0
        high_val = 255
        return spaces.Box(low=low_val, high=high_val,
                          shape=self.stacked_shape, dtype=jnp.uint8)

    def action_space(self, params):
        return self.env.action_space(params)

    def default_params(self):
        return self.env.default_params()

    def reset(self, key, params):
        ob, state = self.env.reset(key, params)
        # We'll store the last frames in 'frame_stacks' inside state
        # Initialize them all to the first observation
        frame_stacks = jnp.tile(ob, (self.num_stack,))
        # We combine 'env_state' + 'frame_stacks' in a dict
        new_state = {"env_state": state, "frames": frame_stacks}
        # Construct stacked obs
        stacked_obs = frame_stacks.reshape(self.stacked_shape)
        return (stacked_obs, new_state)

    def step(self, key, state, action, params):
        # Unpack
        env_state = state["env_state"]
        frames_old = state["frames"]

        ob, env_state_new, reward, done, info = self.env.step(key, env_state, action, params)

        # Shift frames by one, add the new observation
        # frames_old is shape (H, W, C * num_stack) flattened, so let's reshape:
        frames_old_reshaped = frames_old.reshape(self.num_stack, *self.obs_shape)
        # Drop oldest (index=0), keep the rest, then append new frame
        frames_new = jnp.concatenate([frames_old_reshaped[1:], ob[None]], axis=0)
        # Flatten again
        frames_new_flat = frames_new.reshape(-1)
        new_state = {"env_state": env_state_new, "frames": frames_new_flat}

        stacked_obs = frames_new.reshape(self.stacked_shape)
        return (stacked_obs, new_state, reward, done, info)


# ------------------------------------------------------
# 4) Transition NamedTuple
# ------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ------------------------------------------------------
# 5) Rollout + PPO Update Function
# ------------------------------------------------------
def make_rollout_and_update_fn(env, env_params, network, config):
    @jax.jit
    def rollout_and_update(train_state, env_state, last_obs, rng):
        # 1) Collect rollout
        def env_step_fn(carry, _):
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
        carry_out, traj_batch = jax.lax.scan(
            env_step_fn, carry_init, None, length=config["NUM_STEPS"]
        )
        train_state, env_state, last_obs, rng = carry_out

        # 2) GAE advantage
        _, last_val = network.apply(train_state.params, last_obs)

        def gae_scan_fn(carry, transition):
            gae_, next_val = carry
            delta = (
                transition.reward
                + config["GAMMA"] * next_val * (1.0 - transition.done)
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

        # 3) PPO update (minibatching)
        def ppo_update(train_state_, traj_batch_, advantages_, returns_, rng_):
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            minibatch_size = batch_size // config["NUM_MINIBATCHES"]

            # Flatten
            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                traj_batch_,
            )
            adv_flat = advantages_.reshape((batch_size,))
            ret_flat = returns_.reshape((batch_size,))

            rng_, perm_rng = jax.random.split(rng_)
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

            def update_minibatch(train_state_inner, minibatch):
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
                    loss_val = (
                        policy_loss
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return loss_val, (policy_loss, value_loss, entropy)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(train_state_inner.params, traj_mb, adv_mb, ret_mb)
                train_state_inner = train_state_inner.apply_gradients(grads=grads)
                return train_state_inner, loss_val

            def scan_minibatch_fn(train_state_inner, i):
                t_mb = jax.tree_util.tree_map(lambda x: x[i], traj_shuf)
                ga_mb = adv_shuf[i]
                rt_mb = ret_shuf[i]
                train_state_inner, _ = update_minibatch(train_state_inner, (t_mb, ga_mb, rt_mb))
                return train_state_inner, None

            # Multiple epochs
            for _ in range(config["UPDATE_EPOCHS"]):
                indices = jnp.arange(config["NUM_MINIBATCHES"])
                train_state_, _ = jax.lax.scan(scan_minibatch_fn, train_state_, indices)

            return train_state_, rng_

        train_state, rng = ppo_update(train_state, traj_batch, advantages, returns, rng)
        return train_state, env_state, last_obs, rng, traj_batch

    return rollout_and_update


# ------------------------------------------------------
# 6) Deterministic Evaluation
# ------------------------------------------------------
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


# ------------------------------------------------------
# 7) Main Training Loop
# ------------------------------------------------------
def run_ppo_training(config):
    """
    Maintains the same structure as the original ppo_minigrid_wandb_cnn.py,
    but uses AtariCNNActorCritic + optional framestacking for image-based envs.
    """
    # 1) Create environment
    #    If you want to run a "TabularMDP" from an NPZ file, do so.
    #    Otherwise, assume you have an Atari or image-based env. We'll do Breakout as example.
    if config["ENV_NAME"] == "TabularMDP":
        from gymnax_env import TabularEnv  # or your custom class
        env = TabularEnv(config["ENV_FILE"])
        env_params = env.default_params().replace(reward_scale=config["REWARD_SCALE"])
    else:
        # E.g. "Breakout-MinAtar" or standard Gymnax "Atari-Breakout"
        env, env_params = gymnax.make(config["ENV_NAME"])

    # 2) Optionally wrap environment with framestack if requested
    if config.get("FRAMESTACK", 1) > 1:
        env = FrameStackEnv(env, config["FRAMESTACK"])

    # 3) Wrap for logging
    env = LogWrapper(env)

    # 4) Build the correct network
    obs_shape = env.observation_space(env_params).shape  # e.g. (84, 84, 4*FRAMESTACK)
    action_dim = env.action_space(env_params).n

    # If we detect a typical 3D shape => use AtariCNN
    if len(obs_shape) == 3:
        # Use the AtariCNN for image-based env
        network = AtariCNNActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    else:
        # Fallback MLP
        network = MLPActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

    # 5) Initialize parameters & optimizer
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # 6) Reset vector of envs
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    # 7) Build rollout + update function
    rollout_and_update_fn = make_rollout_and_update_fn(env, env_params, network, config)

    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    global_env_steps = 0

    # Track returns for median stats
    train_returns_buffer = []
    best_eval_return = -1e9
    best_median_train = -1e9

    # Main loop
    for update_i in range(config["NUM_UPDATES"]):
        train_state, env_state, obsv, rng, traj_batch = rollout_and_update_fn(
            train_state, env_state, obsv, rng
        )
        global_env_steps += steps_per_update

        # Gather episode returns from info
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


# ------------------------------------------------------
# 8) Script Entry Point
# ------------------------------------------------------
if __name__ == "__main__":
    config = {
        "SEED": 1,
        "LR": 2.5e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,       # typical PPO
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/fruitbot_easy_l0_40_fs8/consolidated.npz",  # only used if ENV_NAME == "TabularMDP"
        "REWARD_SCALE": 1.0,
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 5.0,  # Example threshold
        # Frame stack
        "FRAMESTACK": 4,          # set to 1 to disable, or 4 to enable 4-frame stacking
    }
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // steps_per_update)

    wandb.init(project="ppo_atari_framestack", config=config)
    run_ppo_training(config)
    wandb.finish()
