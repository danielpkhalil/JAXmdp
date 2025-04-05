import os
import jax
import jax.numpy as jnp

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import gymnax
import flashbax as fbx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Q-Networks
# ------------------------------------------------------------------------------

# CNN-based Q-Network for image observations
class MiniGridCNNQNetwork(nn.Module):
    action_dim: int
    features_dim: int = 512
    normalized_image: bool = False  # set to True if images are already normalized
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Assume x has shape (batch, H, W, C)
        if not self.normalized_image and x.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(self.features_dim)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.action_dim)(x)
        return q_values

# MLP Q-Network for vector observations
class MLPQNetwork(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

# ------------------------------------------------------------------------------
# Data structures and Train State
# ------------------------------------------------------------------------------

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

# ------------------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------------------

def make_train(config):

    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"])

    # Environment creation: if using a custom TabularMDP, import and use it;
    # otherwise, use gymnax.make
    if config["ENV_NAME"] == "TabularMDP":
        from gymnax_env import TabularEnv, TabularEnvParams
        basic_env = TabularEnv(config["ENV_FILE"])
        env_params = basic_env.default_params().replace(
            reward_scale=config.get("REWARD_SCALE", 1.0)
        )
        env = LogWrapper(basic_env)
    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"])
        obs_shape = basic_env.observation_space(env_params).shape
        if len(obs_shape) == 3:
            # For image observations, do not flatten.
            env = LogWrapper(basic_env)
        else:
            env = FlattenObservationWrapper(LogWrapper(basic_env))

    # Create batched reset and step functions.
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        # Initialize the environment
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # Initialize the replay buffer.
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
        # Use a dummy RNG for buffer initialization.
        dummy_rng = jax.random.PRNGKey(42)
        _action = basic_env.action_space(env_params).sample(dummy_rng)
        _, _env_state = env.reset(dummy_rng, env_params)
        _obs, _, _reward, _done, _ = env.step(dummy_rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # Initialize the Q-network.
        action_dim = basic_env.action_space(env_params).n
        obs_shape = basic_env.observation_space(env_params).shape
        if len(obs_shape) == 3:
            # Use the CNN network for image observations.
            network = MiniGridCNNQNetwork(action_dim=action_dim, normalized_image=False)
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
        else:
            network = MLPQNetwork(action_dim=action_dim)
            dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, dummy_obs)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_util.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # Epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(rng, 2)
            eps = jnp.clip(
                ((config["EPSILON_FINISH"] - config["EPSILON_START"]) / config["EPSILON_ANNEAL_TIME"]) * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)
            chosen_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape) < eps,
                jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
                greedy_actions,
            )
            return chosen_actions

        # Training loop update step.
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state

            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(rng_a, q_vals, train_state.timesteps)
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(rng_s, env_state, action)
            train_state = train_state.replace(timesteps=train_state.timesteps + config["NUM_ENVS"])

            # Update buffer.
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # Network update phase.
            def _learn_phase(train_state, rng):
                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(train_state.target_network_params, learn_batch.second.obs)
                q_next_target = jnp.max(q_next_target, axis=-1)
                target = learn_batch.first.reward + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target

                def _loss_fn(params):
                    q_vals = network.apply(params, learn_batch.first.obs)
                    chosen_q_vals = jnp.take_along_axis(
                        q_vals, jnp.expand_dims(learn_batch.first.action, axis=-1), axis=-1
                    ).squeeze(-1)
                    return jnp.mean((chosen_q_vals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
                & (train_state.timesteps > config["LEARNING_STARTS"])
                & (train_state.timesteps % config["TRAINING_INTERVAL"] == 0)
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda ts, rng: _learn_phase(ts, rng),
                lambda ts, rng: (ts, jnp.array(0.0)),
                train_state,
                _rng,
            )

            # Update target network.
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(ts.params, ts.target_network_params, config["TAU"])
                ),
                lambda ts: ts,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

            if config.get("WANDB_MODE", "disabled") == "online":
                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)
                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

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
        "ENV_NAME": "TabularMDP",  # set to "TabularMDP" or your other env name
        "ENV_FILE": "/nas/ucb/cassidy/rl-theory/data/mdps/fruitbot_easy_l0_40_fs8/consolidated.npz",  # if using TabularMDP
        "REWARD_SCALE": 1.0,
        "SEED": 0,
        "NUM_SEEDS": 5,
        "WANDB_MODE": "disabled",  # change to "online" to log to wandb
    }

    wandb.init(
        entity="",
        project="",
        tags=["DQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'purejaxrl_dqn_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    returns = jax.device_get(outs['metrics']['returns'])
    timesteps = jax.device_get(outs['metrics']['timesteps'])

    num_seeds = returns.shape[0]

    plt.figure(figsize=(10, 6))
    for i in range(num_seeds):
        plt.plot(timesteps[i], returns[i], label=f"Seed {i}")
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("Reward Curves over Timesteps for Each Seed")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
