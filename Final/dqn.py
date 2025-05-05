import os
import csv
import time
import numpy as np
import jax
import jax.numpy as jnp
import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState as FlaxTrainState
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import gymnax
import flashbax as fbx

# ------------------------------
# Custom Train State & TimeStep
# ------------------------------
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

# ------------------------------
# Q-Network Architectures
# ------------------------------
class MiniGridCNNQNetwork(nn.Module):
    action_dim: int
    features_dim: int = 512
    normalized_image: bool = False
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if not self.normalized_image and x.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=32, kernel_size=(3,3), strides=(2,2), padding="SAME")(x); x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(2,2), padding="SAME")(x); x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(1,1), padding="SAME")(x); x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.features_dim)(x); x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

class MLPQNetwork(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(120)(x); x = nn.relu(x)
        x = nn.Dense(84)(x);  x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

# ------------------------------
# Build train functions
# ------------------------------
def make_train(config):
    # Compute number of update steps
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"])
    # Environment setup
    if config["ENV_NAME"] == "TabularMDP":
        from gymnax_env import TabularEnv, TabularEnvParams
        basic_env = TabularEnv(config["ENV_FILE"])
        env_params = basic_env.default_params().replace(reward_scale=config.get("REWARD_SCALE",1.0))
        env = LogWrapper(basic_env)
    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"])
        obs_shape = basic_env.observation_space(env_params).shape
        if len(obs_shape) == 3:
            env = LogWrapper(basic_env)
        else:
            env = FlattenObservationWrapper(LogWrapper(basic_env))

    # Vectorized helpers
    vmap_reset = lambda n: lambda rng: jax.vmap(env.reset, in_axes=(0,None))(
        jax.random.split(rng, n), env_params
    )
    vmap_step  = lambda n: lambda rng, s, a: jax.vmap(env.step, in_axes=(0,0,0,None))(
        jax.random.split(rng, n), s, a, env_params
    )

    # Build network & optimizer
    action_dim = basic_env.action_space(env_params).n
    obs_shape  = basic_env.observation_space(env_params).shape
    if len(obs_shape) == 3:
        net = MiniGridCNNQNetwork(action_dim, normalized_image=False)
        dummy_obs = jnp.zeros((1,)+obs_shape, dtype=jnp.uint8)
    else:
        net = MLPQNetwork(action_dim)
        dummy_obs = jnp.zeros((1,)+obs_shape, dtype=jnp.float32)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, dummy_obs)
    lr = (lambda count: config["LR"] * (1.0 - count/config["NUM_UPDATES"])) \
         if config.get("LR_LINEAR_DECAY",False) else config["LR"]
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
        max_length=config["BUFFER_SIZE"],
        min_length=config["BUFFER_BATCH_SIZE"],
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_sequences=False,
        add_batch_size=config["NUM_ENVS"],
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    # Initialize buffer with a dummy transition
    dummy_rng = jax.random.PRNGKey(42)
    a0 = basic_env.action_space(env_params).sample(dummy_rng)
    _, s0 = env.reset(dummy_rng, env_params)
    o0, _, r0, d0, _ = env.step(dummy_rng, s0, a0, env_params)
    ts0 = TimeStep(obs=o0, action=a0, reward=r0, done=d0)
    buffer_state = buffer.init(ts0)

    # ε-greedy policy
    def eps_greedy(rng, qv, t):
        rng1, rng2 = jax.random.split(rng,2)
        eps = jnp.clip(
            ((config["EPSILON_FINISH"]-config["EPSILON_START"])/config["EPSILON_ANNEAL_TIME"])*t
            + config["EPSILON_START"],
            config["EPSILON_FINISH"],
        )
        greedy = jnp.argmax(qv, axis=-1)
        return jnp.where(
            jax.random.uniform(rng2, greedy.shape)<eps,
            jax.random.randint(rng1, greedy.shape, 0, qv.shape[-1]),
            greedy,
        )

    # Single‐step update
    def _update(runner, _):
        ts, buf_st, env_st, last_obs, rng = runner
        rng, r1, r2, r3 = jax.random.split(rng,4)
        qv = net.apply(ts.params, last_obs)
        a = eps_greedy(r1, qv, ts.timesteps)
        obs, env_st, rew, done, info = vmap_step(config["NUM_ENVS"])(r2, env_st, a)
        ts = ts.replace(timesteps=ts.timesteps+config["NUM_ENVS"])
        buf_st = buffer.add(buf_st, TimeStep(obs=last_obs, action=a, reward=rew, done=done))

        # learning phase
        def learn_phase(state, rng_in):
            batch = buffer.sample(buf_st, rng_in).experience
            q_next = net.apply(state.target_network_params, batch.second.obs)
            q_next = jnp.max(q_next, axis=-1)
            target = batch.first.reward + (1-batch.first.done)*config["GAMMA"]*q_next
            def loss_fn(p):
                qv2 = net.apply(p, batch.first.obs)
                chosen = jnp.take_along_axis(
                    qv2, jnp.expand_dims(batch.first.action,-1), -1
                ).squeeze(-1)
                return jnp.mean((chosen-target)**2)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grads).replace(
                n_updates=state.n_updates+1
            )
            return new_state, loss

        do_learn = (
            buffer.can_sample(buf_st)
            & (ts.timesteps>config["LEARNING_STARTS"])
            & (ts.timesteps % config["TRAINING_INTERVAL"]==0)
        )
        ts, loss = jax.lax.cond(
            do_learn,
            lambda st,r: learn_phase(st,r),
            lambda st,r: (st, jnp.array(0.0)),
            ts, r3
        )
        ts = jax.lax.cond(
            ts.timesteps % config["TARGET_UPDATE_INTERVAL"]==0,
            lambda st: st.replace(
                target_network_params=optax.incremental_update(
                    st.params, st.target_network_params, config["TAU"]
                )
            ),
            lambda st: st,
            ts
        )

        metrics = {
            "timesteps": ts.timesteps,
            "loss":      loss.mean(),
            "train_return": info["returned_episode_returns"].mean(),
        }
        # internal W&B callback every 100 steps
        if config.get("WANDB_MODE","disabled")=="online":
            def cb(m):
                if m["timesteps"] % 100 == 0:
                    wandb.log(m, step=int(m["timesteps"]))
            jax.debug.callback(cb, metrics)

        return (ts, buf_st, env_st, obs, rng), metrics

    update_fn = jax.jit(lambda runner: _update(runner, None))

    # Initial runner state
    def init_fn(rng_in):
        rng1, rng2 = jax.random.split(rng_in)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(rng1)
        runner = (train_state, buffer_state, env_state, init_obs, rng2)
        return runner

    return init_fn, update_fn, net, env, env_params

# ------------------------------
# Deterministic Evaluation
# ------------------------------
def evaluate_det_q(train_state, net, env, env_params, rng, max_steps=10000):
    obs, state = env.reset(rng, env_params)
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        qv = net.apply(train_state.params, obs[None, ...])[0]
        action = int(jnp.argmax(qv))
        obs, state, reward, done, info = env.step(rng, state, action, env_params)
        total_reward += float(reward)
        steps += 1
    return total_reward

# ------------------------------
# Main: Training + Early Stop
# ------------------------------
def main():
    config = {
        "NUM_ENVS": 10,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "TabularMDP",
        "ENV_FILE": "test.npz",
        "REWARD_SCALE": 1.0,
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",
        # New eval/stop params
        "EVAL_FREQUENCY": 1000,
        "TRAIN_MEDIAN_WINDOW": 20,
        "OPTIMAL_REWARD": 20.0,
    }

    wandb.init(project="DQN", config=config, mode=config["WANDB_MODE"])
    init_fn, update_fn, net, env, env_params = make_train(config)

    # Prepare CSV logging
    csv_file = open("dqn_metrics.csv", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "seed", "update", "timesteps", "loss",
        "train_return", "median_train_return",
        "eval_return", "best_eval_return"
    ])
    writer.writeheader()

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    overall_t0 = time.time()
    for seed_idx, seed_rng in enumerate(rngs):
        runner = init_fn(seed_rng)
        train_buf = []
        best_eval = -1e9
        best_med_train = -1e9

        for u in range(config["NUM_UPDATES"]):
            # Step
            runner, metrics = update_fn(runner)
            t = int(metrics["timesteps"])
            loss_val = float(metrics["loss"])
            tr_ret = float(metrics["train_return"])
            train_buf.append(tr_ret)

            # Periodic evaluation
            eval_ret = None
            prev_t = t - config["NUM_ENVS"]
            if (t // config["EVAL_FREQUENCY"]) != (prev_t // config["EVAL_FREQUENCY"]):
                eval_rng = jax.random.split(runner[4], 1)[0]
                eval_ret = evaluate_det_q(runner[0], net, env, env_params, eval_rng)
                best_eval = max(best_eval, eval_ret)

            # Median of recent training returns
            recent = train_buf[-config["TRAIN_MEDIAN_WINDOW"]:]
            med_train = float(np.median(recent)) if recent else 0.0
            best_med_train = max(best_med_train, med_train)

            # Log to CSV & W&B
            row = {
                "seed": seed_idx, "update": u, "timesteps": t, "loss": loss_val,
                "train_return": tr_ret, "median_train_return": med_train,
                "eval_return": eval_ret if eval_ret is not None else "",
                "best_eval_return": best_eval,
            }
            writer.writerow(row)
            wandb.log({
                "update": u,
                "timesteps": t,
                "loss": loss_val,
                "train_return": tr_ret,
                "median_train_return": med_train,
                "eval_return": eval_ret if eval_ret is not None else np.nan,
                "best_eval_return": best_eval,
            }, step=t)

            # Early stopping
            if max(best_eval, best_med_train) >= config["OPTIMAL_REWARD"]:
                print(
                    f"[Seed {seed_idx}] Optimal reward {config['OPTIMAL_REWARD']} reached "
                    f"at update {u}, timesteps {t}"
                )
                break

    total_time = time.time() - overall_t0
    print(f"Training complete in {total_time:.1f}s")
    csv_file.close()

if __name__ == "__main__":
    main()

