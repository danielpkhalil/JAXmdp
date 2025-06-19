"""
dqn.py – vanilla JAX DQN (flashbax replay) with minimal changes to
support action-mask observations from TabularEnv.
"""

import csv
import os
import time
import pprint
from typing import TypedDict

import chex
import flashbax as fbx
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState as FlaxTrainState
from gymnax.wrappers.purerl import LogWrapper
from gymnax.environments import spaces                    # <<< CHANGED
import jax.lax as lax                                     # <<< CHANGED


# ------------------------------------------------------------------------
# helper: mask invalid Q-values
# ------------------------------------------------------------------------
def _mask_q(q_values: jnp.ndarray, mask: jnp.ndarray):     # --- NEW
    return q_values + (1.0 - mask) * -1e9                 # -inf for invalid


class DQNArgs(TypedDict):
    num_envs: int
    buffer_size: int
    buffer_batch_size: int
    epsilon_start: float
    epsilon_finish: float
    epsilon_anneal_time: int
    target_update_interval: int
    lr: float
    learning_starts: int
    training_interval: int
    lr_linear_decay: bool
    tau: float
    gamma: float


# ------------------------------ dataclasses ------------------------------
@chex.dataclass
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(FlaxTrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


# ------------------------------ networks ---------------------------------
class MiniGridCNNQ(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):                                 # <<< CHANGED
        if isinstance(x, dict):                            # <<< CHANGED
            mask = x["action_mask"]                        # <<< CHANGED
            x = x["obs"]                                   # <<< CHANGED
        else:                                              # <<< CHANGED
            mask = jnp.ones((x.shape[0], self.action_dim), jnp.float32)  # <<< CHANGED

        x = x.astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, (3, 3), (2, 2), "SAME")(x))
        x = nn.relu(nn.Conv(64, (3, 3), (2, 2), "SAME")(x))
        x = nn.relu(nn.Conv(64, (3, 3), (1, 1), "SAME")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(512)(x))
        q_raw = nn.Dense(self.action_dim)(x)
        return _mask_q(q_raw, mask)                        # <<< CHANGED


class MLPQ(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):                                 # <<< CHANGED
        if isinstance(x, dict):                            # <<< CHANGED
            mask = x["action_mask"]; x = x["obs"]          # <<< CHANGED
        else:                                              # <<< CHANGED
            mask = jnp.ones((x.shape[0], self.action_dim), jnp.float32)  # <<< CHANGED

        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        q_raw = nn.Dense(self.action_dim)(x)
        return _mask_q(q_raw, mask)                        # <<< CHANGED


# ------------------------------
# Build train functions
# ------------------------------
def make_train(env, env_params, algo_args: DQNArgs, seed: int, num_updates):
    # Vectorized helpers
    vmap_reset = lambda n: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n), env_params
    )
    vmap_step = lambda n: lambda rng, s, a: jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(rng, n), s, a, env_params
    )

    # Wrap env
    env = LogWrapper(env)

    # Build network & optimizer
    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    # dummy observation + network selection -------------------------------
    if isinstance(obs_space, spaces.Dict):  # <<< CHANGED
        base = obs_space.spaces["obs"]
        dummy_obs = {
            "obs": jnp.zeros((1,) + base.shape, dtype=base.dtype),
            "action_mask": jnp.ones((1, action_dim), jnp.float32),
        }
        uses_cnn = len(base.shape) == 3
    else:
        dummy_obs = jnp.zeros((1,) + obs_space.shape, dtype=obs_space.dtype)
        uses_cnn = len(obs_space.shape) == 3

    net = (MiniGridCNNQ if uses_cnn else MLPQ)(action_dim)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, dummy_obs)
    lr = (
        (lambda count: algo_args["lr"] * (1.0 - count / num_updates))
        if algo_args.get("lr_linear_decay", False)
        else algo_args["lr"]
    )
    tx = optax.adam(learning_rate=lr)
    train_state = CustomTrainState.create(
        apply_fn=net.apply,
        params=params,
        target_network_params=jax.tree_util.tree_map(lambda x: x, params),
        tx=tx,
        timesteps=0,
        n_updates=0,
    )

    # Replay buffer setup
    buffer = fbx.make_flat_buffer(
        max_length=algo_args["buffer_size"],
        min_length=algo_args["buffer_batch_size"],
        sample_batch_size=algo_args["buffer_batch_size"],
        add_sequences=False,
        add_batch_size=algo_args["num_envs"],
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    # Initialize buffer with a dummy transition
    dummy_rng = jax.random.PRNGKey(42)
    a0 = env.action_space(env_params).sample(dummy_rng)
    _, s0 = env.reset(dummy_rng, env_params)
    o0, _, r0, d0, _ = env.step(dummy_rng, s0, a0, env_params)
    ts0 = TimeStep(obs=o0, action=a0, reward=r0, done=d0)
    buffer_state = buffer.init(ts0)

    # ε-greedy policy
    def eps_greedy(rng, qv, t):
        rng1, rng2 = jax.random.split(rng, 2)
        eps = jnp.clip(
            (
                (algo_args["epsilon_finish"] - algo_args["epsilon_start"])
                / algo_args["epsilon_anneal_time"]
            )
            * t
            + algo_args["epsilon_start"],
            algo_args["epsilon_finish"],
        )
        greedy = jnp.argmax(qv, axis=-1)
        return jnp.where(
            jax.random.uniform(rng2, greedy.shape) < eps,
            jax.random.randint(rng1, greedy.shape, 0, qv.shape[-1]),
            greedy,
        )

    # Single‐step update
    def _update(runner, _):
        ts, buf_st, env_st, last_obs, rng = runner
        rng, r1, r2, r3 = jax.random.split(rng, 4)
        qv = net.apply(ts.params, last_obs)
        a = eps_greedy(r1, qv, ts.timesteps)
        obs, env_st, rew, done, info = vmap_step(algo_args["num_envs"])(r2, env_st, a)
        ts = ts.replace(timesteps=ts.timesteps + algo_args["num_envs"])
        buf_st = buffer.add(
            buf_st, TimeStep(obs=last_obs, action=a, reward=rew, done=done)
        )

        # learning phase
        def learn_phase(state, rng_in):
            batch = buffer.sample(buf_st, rng_in).experience
            q_next = net.apply(state.target_network_params, batch.second.obs)
            q_next = jnp.max(q_next, axis=-1)
            target = (
                batch.first.reward + (1 - batch.first.done) * algo_args["gamma"] * q_next
            )

            def loss_fn(p):
                qv2 = net.apply(p, batch.first.obs)
                chosen = jnp.take_along_axis(
                    qv2, jnp.expand_dims(batch.first.action, -1), -1
                ).squeeze(-1)
                return jnp.mean((chosen - target) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grads).replace(
                n_updates=state.n_updates + 1
            )
            return new_state, loss

        do_learn = (
            buffer.can_sample(buf_st)
            & (ts.timesteps > algo_args["learning_starts"])
            & (ts.timesteps % algo_args["training_interval"] == 0)
        )
        ts, loss = jax.lax.cond(
            do_learn,
            lambda st, r: learn_phase(st, r),
            lambda st, r: (st, jnp.array(0.0)),
            ts,
            r3,
        )
        ts = jax.lax.cond(
            ts.timesteps % algo_args["target_update_interval"] == 0,
            lambda st: st.replace(
                target_network_params=optax.incremental_update(
                    st.params, st.target_network_params, algo_args["tau"]
                )
            ),
            lambda st: st,
            ts,
        )

        metrics = {
            "timesteps": ts.timesteps,
            "loss": loss.mean(),
            "train_return": info["returned_episode_returns"].mean(),
        }
        # # internal W&B callback every 100 steps
        # if algo_args.get("WANDB_MODE", "disabled") == "online":
        #
        #     def cb(m):
        #         if m["timesteps"] % 100 == 0:
        #             wandb.log(m, step=int(m["timesteps"]))
        #
        #     jax.debug.callback(cb, metrics)

        return (ts, buf_st, env_st, obs, rng), metrics

    update_fn = jax.jit(lambda runner: _update(runner, None))

    # Initial runner state
    def init_fn(rng_in):
        rng1, rng2 = jax.random.split(rng_in)
        init_obs, env_state = vmap_reset(algo_args["num_envs"])(rng1)
        runner = (train_state, buffer_state, env_state, init_obs, rng2)
        return runner

    return init_fn, update_fn, net, env, env_params


# ------------------------------ evaluation --------------------------------
def evaluate_policy_deterministic(train_state, q_net, env, env_params, rng, max_steps=10000):
    obs, state = env.reset(rng, env_params)
    total_reward, steps, done = 0.0, 0, False
    while not done and steps < max_steps:
        q_values = q_net.apply(                               # <<< CHANGED
            train_state.params,
            {"obs": obs["obs"][None, ...],
             "action_mask": obs["action_mask"][None, ...]} if isinstance(obs, dict)
            else obs[None, ...],
        )[0]
        action = int(jnp.argmax(q_values))
        obs, state, reward, done, _ = env.step(rng, state, action, env_params)
        total_reward += float(reward); steps += 1
    return total_reward


# ------------------------------
# Training + Early Stop
# ------------------------------
def run_dqn(
    env,
    env_params,
    total_timesteps,
    seed,
    eval_frequency,
    stop_on_eval_reward,
    stop_on_median_train_reward,
    algo_args: DQNArgs,
    log_dir: str,
):
    steps_per_update = algo_args["num_envs"]
    num_updates = int(total_timesteps // steps_per_update)

    # build training functions
    init_fn, update_fn, net, env, env_params = make_train(env, env_params, algo_args, seed, num_updates)

    # prepare CSV logging
    csv_file = open(os.path.join(log_dir, "metrics.csv"), "w", newline="")
    fieldnames = [
        "update",
        "timesteps",
        "loss",
        "train_return",
        "median_train_return",
        "eval_return",
        "best_eval_return",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # run training loop
    rng = jax.random.PRNGKey(seed)
    overall_t0 = time.time()

    runner = init_fn(rng)
    train_buf = []
    best_eval = -1e9
    best_med_train = -1e9

    for u in range(num_updates):
        runner, metrics = update_fn(runner)
        t = int(metrics["timesteps"])
        tr_ret = float(metrics["train_return"])
        train_buf.append(tr_ret)

        # evaluation
        eval_ret = None
        prev_t = t - algo_args["num_envs"]
        if (t // eval_frequency) != (prev_t // eval_frequency):
            eval_rng = jax.random.split(runner[4], 1)[0]
            eval_ret = evaluate_policy_deterministic(runner[0], net, env, env_params, eval_rng)
            best_eval = max(best_eval, eval_ret)

        # median train returns
        recent = train_buf[-100 :]
        med_train = float(np.median(recent)) if recent else 0.0
        best_med_train = max(best_med_train, med_train)

        # log
        csv_writer.writerow(
            {
                "update": u,
                "timesteps": t,
                "loss": float(metrics["loss"]),
                "train_return": tr_ret,
                "median_train_return": med_train,
                "eval_return": eval_ret if eval_ret is not None else "",
                "best_eval_return": best_eval,
            }
        )
        pprint.pp(metrics)
        print()

        # early stopping
        if eval_ret is not None and eval_ret >= stop_on_eval_reward:
            print(f"Reached eval reward {eval_ret:.2f}")
            break
        elif med_train >= stop_on_median_train_reward:
            print(f"Reached median train reward {med_train:.2f}")
            break

    total_time = time.time() - overall_t0
    print(f"DQN training complete in {total_time:.1f}s")
    csv_file.close()