import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

# Import your custom TabularEnv creation function and wrappers
from gymnax_env import create_tabular_env
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

# ------------------------------------------------------------------------------
# Actor-Critic Network Definition
# ------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        # If the input is an image (Box observation), cast to float,
        # normalize to [0, 1], and flatten.
        if x.ndim > 2:
            x = x.astype(jnp.float32) / 255.0
            x = x.reshape((x.shape[0], -1))
        # Choose the activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        # --- Actor network ---
        actor_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_hidden = activation_fn(actor_hidden)
        actor_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_hidden)
        actor_hidden = activation_fn(actor_hidden)
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_hidden)
        pi = distrax.Categorical(logits=logits)

        # --- Critic network ---
        critic_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic_hidden = activation_fn(critic_hidden)
        critic_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_hidden)
        critic_hidden = activation_fn(critic_hidden)
        critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_hidden
        )
        critic_value = jnp.squeeze(critic_value, axis=-1)

        return pi, critic_value

# ------------------------------------------------------------------------------
# Transition NamedTuple for storing trajectories
# ------------------------------------------------------------------------------

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any

# ------------------------------------------------------------------------------
# PPO Training Function
# ------------------------------------------------------------------------------

def make_train(config):
    # Derived configurations
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create the custom TabularEnv
    env = create_tabular_env("consolidated.npz")
    # If using screen observations, do NOT flatten them here so the network can process images.
    # env = FlattenObservationWrapper(env)  # Not used for image observations.
    env = LogWrapper(env)
    env_params = env.default_params

    # Determine observation shape and action dimension.
    obs_shape = env.observation_space(env_params).shape  # For Box observations.
    action_dim = env.action_space(env_params).n

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # --- Initialize the network ---
        network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        # For Box observations (screens), initialize with the correct shape.
        init_x = jnp.zeros((config["NUM_ENVS"],) + obs_shape, dtype=jnp.uint8)
        network_params = network.init(_rng, init_x)

        # --- Set up the optimizer ---
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
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # --- Initialize the environment ---
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_rng, env_params)

        # --- Training loop ---
        def _update_step(runner_state, unused):
            # Collect trajectories over NUM_STEPS time steps.
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select action using the current policy.
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Step the environment.
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step_env, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            # Compute advantages using Generalized Advantage Estimation (GAE).
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Update the network with collected trajectories.
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, gae, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        ) * gae_norm
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux_vals), grads = grad_fn(
                        train_state.params, traj_batch, gae, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, gae, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], "batch size must equal num_steps * num_envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, gae, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, gae, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info  # You can extract further metrics here.
            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1, #4
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 1, #4
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
    }

    rng = jax.random.PRNGKey(30)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    out = train_jit(rng)
    print("Training finished. Metrics:")
    print(out["metrics"])










# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import numpy as np
# import optax
# from flax.linen.initializers import constant, orthogonal
# from typing import Sequence, NamedTuple, Any
# from flax.training.train_state import TrainState
# import distrax
#
# # Import your custom TabularEnv creation function
# from gymnax_env import create_tabular_env
# from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
#
# class ActorCritic(nn.Module):
#     action_dim: Sequence[int]
#     activation: str = "tanh"
#
#     @nn.compact
#     def __call__(self, x):
#         if self.activation == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh
#         actor_mean = nn.Dense(
#             64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(x)
#         actor_mean = activation(actor_mean)
#         actor_mean = nn.Dense(
#             64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(actor_mean)
#         actor_mean = activation(actor_mean)
#         actor_mean = nn.Dense(
#             self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(actor_mean)
#         pi = distrax.Categorical(logits=actor_mean)
#
#         critic = nn.Dense(
#             64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(x)
#         critic = activation(critic)
#         critic = nn.Dense(
#             64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(critic)
#         critic = activation(critic)
#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
#             critic
#         )
#
#         return pi, jnp.squeeze(critic, axis=-1)
#
#
# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     info: jnp.ndarray
#
#
# def make_train(config):
#     # Derived configs
#     config["NUM_UPDATES"] = (
#         config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
#     )
#     config["MINIBATCH_SIZE"] = (
#         config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
#     )
#
#     # Create the custom TabularEnv
#     env = create_tabular_env("consolidated.npz")
#     #env = FlattenObservationWrapper(env)
#     env = LogWrapper(env)
#     env_params = env.default_params
#
#     def linear_schedule(count):
#         frac = (
#             1.0
#             - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
#             / config["NUM_UPDATES"]
#         )
#         return config["LR"] * frac
#
#     def train(rng):
#         # INIT NETWORK
#         network = ActorCritic(
#             env.action_space(env_params).n, activation=config["ACTIVATION"]
#         )
#         rng, _rng = jax.random.split(rng)
#         init_x = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32) #init_x = jnp.zeros(env.observation_space(env_params).shape)
#         network_params = network.init(_rng, init_x)
#
#         if config["ANNEAL_LR"]:
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(learning_rate=linear_schedule, eps=1e-5),
#             )
#         else:
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(config["LR"], eps=1e-5),
#             )
#
#         train_state = TrainState.create(
#             apply_fn=network.apply,
#             params=network_params,
#             tx=tx,
#         )
#
#         # INIT ENV
#         rng, _rng = jax.random.split(rng)
#         reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
#         obsv, env_state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_rng, env_params)
#
#         # TRAIN LOOP
#         def _update_step(runner_state, unused):
#             # COLLECT TRAJECTORIES
#             def _env_step(runner_state, unused):
#                 train_state, env_state, last_obs, rng = runner_state
#
#                 # SELECT ACTION
#                 rng, _rng = jax.random.split(rng)
#                 pi, value = network.apply(train_state.params, last_obs)
#                 action = pi.sample(seed=_rng)
#                 log_prob = pi.log_prob(action)
#
#                 # STEP ENV
#                 rng, _rng = jax.random.split(rng)
#                 rng_step = jax.random.split(_rng, config["NUM_ENVS"])
#                 obsv, env_state, reward, done, info = jax.vmap(
#                     env.step_env, in_axes=(0, 0, 0, None)
#                 )(rng_step, env_state, action, env_params)
#                 transition = Transition(
#                     done, action, value, reward, log_prob, last_obs, info
#                 )
#
#                 runner_state = (train_state, env_state, obsv, rng)
#                 return runner_state, transition
#
#             runner_state, traj_batch = jax.lax.scan(
#                 _env_step, runner_state, None, config["NUM_STEPS"]
#             )
#
#             # CALCULATE ADVANTAGE
#             train_state, env_state, last_obs, rng = runner_state
#             _, last_val = network.apply(train_state.params, last_obs)
#
#             def _calculate_gae(traj_batch, last_val):
#                 def _get_advantages(gae_and_next_value, transition):
#                     gae, next_value = gae_and_next_value
#                     done, value, reward = (
#                         transition.done,
#                         transition.value,
#                         transition.reward,
#                     )
#                     delta = reward + config["GAMMA"] * next_value * (1 - done) - value
#                     gae = (
#                         delta
#                         + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
#                     )
#                     return (gae, value), gae
#
#                 _, advantages = jax.lax.scan(
#                     _get_advantages,
#                     (jnp.zeros_like(last_val), last_val),
#                     traj_batch,
#                     reverse=True,
#                     unroll=16,
#                 )
#                 return advantages, advantages + traj_batch.value
#
#             advantages, targets = _calculate_gae(traj_batch, last_val)
#
#             # UPDATE NETWORK
#             def _update_epoch(update_state, unused):
#                 def _update_minbatch(train_state, batch_info):
#                     traj_batch, advantages, targets = batch_info
#
#                     def _loss_fn(params, traj_batch, gae, targets):
#                         # RERUN NETWORK
#                         pi, value = network.apply(params, traj_batch.obs)
#                         log_prob = pi.log_prob(traj_batch.action)
#
#                         # CALCULATE VALUE LOSS
#                         value_pred_clipped = traj_batch.value + (
#                             value - traj_batch.value
#                         ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
#                         value_losses = jnp.square(value - targets)
#                         value_losses_clipped = jnp.square(value_pred_clipped - targets)
#                         value_loss = (
#                             0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
#                         )
#
#                         # CALCULATE ACTOR LOSS
#                         ratio = jnp.exp(log_prob - traj_batch.log_prob)
#                         gae = (gae - gae.mean()) / (gae.std() + 1e-8)
#                         loss_actor1 = ratio * gae
#                         loss_actor2 = (
#                             jnp.clip(
#                                 ratio,
#                                 1.0 - config["CLIP_EPS"],
#                                 1.0 + config["CLIP_EPS"],
#                             )
#                             * gae
#                         )
#                         loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
#                         loss_actor = loss_actor.mean()
#                         entropy = pi.entropy().mean()
#
#                         total_loss = (
#                             loss_actor
#                             + config["VF_COEF"] * value_loss
#                             - config["ENT_COEF"] * entropy
#                         )
#                         return total_loss, (value_loss, loss_actor, entropy)
#
#                     grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
#                     total_loss, grads = grad_fn(
#                         train_state.params, traj_batch, advantages, targets
#                     )
#                     train_state = train_state.apply_gradients(grads=grads)
#                     return train_state, total_loss
#
#                 train_state, traj_batch, advantages, targets, rng = update_state
#                 rng, _rng = jax.random.split(rng)
#
#                 # Batching and Shuffling
#                 batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
#                 assert (
#                     batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
#                 ), "batch size must be equal to number of steps * number of envs"
#                 permutation = jax.random.permutation(_rng, batch_size)
#                 batch = (traj_batch, advantages, targets)
#                 batch = jax.tree_util.tree_map(
#                     lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
#                 )
#                 shuffled_batch = jax.tree_util.tree_map(
#                     lambda x: jnp.take(x, permutation, axis=0), batch
#                 )
#
#                 # Mini-batch Updates
#                 minibatches = jax.tree_util.tree_map(
#                     lambda x: jnp.reshape(
#                         x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
#                     ),
#                     shuffled_batch,
#                 )
#                 train_state, total_loss = jax.lax.scan(
#                     _update_minbatch, train_state, minibatches
#                 )
#                 update_state = (train_state, traj_batch, advantages, targets, rng)
#                 return update_state, total_loss
#
#             # Updating Training State and Metrics:
#             update_state = (train_state, traj_batch, advantages, targets, rng)
#             update_state, loss_info = jax.lax.scan(
#                 _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
#             )
#             train_state = update_state[0]
#             metric = traj_batch.info
#             rng = update_state[-1]
#
#             # Debugging mode
#             if config.get("DEBUG"):
#                 def callback(info):
#                     return_values = info["returned_episode_returns"][info["returned_episode"]]
#                     timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
#                     for t in range(len(timesteps)):
#                         print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
#                 jax.debug.callback(callback, metric)
#
#             runner_state = (train_state, env_state, last_obs, rng)
#             return runner_state, metric
#
#         rng, _rng = jax.random.split(rng)
#         runner_state = (train_state, env_state, obsv, _rng)
#         runner_state, metric = jax.lax.scan(
#             _update_step, runner_state, None, config["NUM_UPDATES"]
#         )
#         return {"runner_state": runner_state, "metrics": metric}
#
#     return train
#
#
# if __name__ == "__main__":
#     config = {
#         "LR": 2.5e-4,
#         "NUM_ENVS": 4,
#         "NUM_STEPS": 128,
#         "TOTAL_TIMESTEPS": 5e5,
#         "UPDATE_EPOCHS": 4,
#         "NUM_MINIBATCHES": 4,
#         "GAMMA": 0.99,
#         "GAE_LAMBDA": 0.95,
#         "CLIP_EPS": 0.2,
#         "ENT_COEF": 0.01,
#         "VF_COEF": 0.5,
#         "MAX_GRAD_NORM": 0.5,
#         "ACTIVATION": "tanh",
#         # ENV_NAME not used - we've replaced with create_tabular_env
#         "ANNEAL_LR": True,
#         "DEBUG": True,
#     }
#
#     rng = jax.random.PRNGKey(30)
#     train_fn = make_train(config)
#     train_jit = jax.jit(train_fn)
#     out = train_jit(rng)
#
#
#
#
#
#
#
#
#
#
#
#
# # import jax
# # import jax.numpy as jnp
# # import flax.linen as nn
# # import numpy as np
# # import optax
# # from flax.linen.initializers import constant, orthogonal
# # from typing import Sequence, NamedTuple, Any
# # from flax.training.train_state import TrainState
# # import distrax
# #
# # # W&B logging
# # import wandb
# #
# # # Gymnax wrappers for flattening/logging
# # from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
# #
# # # Import your custom environment
# # from gymnax_env import create_tabular_env
# #
# #
# # class ActorCritic(nn.Module):
# #     action_dim: Sequence[int]
# #     activation: str = "tanh"
# #
# #     @nn.compact
# #     def __call__(self, x):
# #         # Select activation function
# #         if self.activation == "relu":
# #             activation = nn.relu
# #         else:
# #             activation = nn.tanh
# #
# #         # Actor network
# #         actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
# #         actor_mean = activation(actor_mean)
# #         actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
# #         actor_mean = activation(actor_mean)
# #         actor_mean = nn.Dense(
# #             self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
# #         )(actor_mean)
# #         pi = distrax.Categorical(logits=actor_mean)
# #
# #         # Critic network
# #         critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
# #         critic = activation(critic)
# #         critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
# #         critic = activation(critic)
# #         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
# #
# #         return pi, jnp.squeeze(critic, axis=-1)
# #
# #
# # class Transition(NamedTuple):
# #     done: jnp.ndarray
# #     action: jnp.ndarray
# #     value: jnp.ndarray
# #     reward: jnp.ndarray
# #     log_prob: jnp.ndarray
# #     obs: jnp.ndarray
# #     info: jnp.ndarray
# #
# #
# # def make_train(config):
# #     """
# #     Create a train function that runs PPO on your custom TabularEnv.
# #     """
# #
# #     # -------------------------------------------------------
# #     # 1. Environment Setup (REPLACE gymnax.make WITH YOUR ENV)
# #     # -------------------------------------------------------
# #     env = create_tabular_env(config["PROBLEM_FILE"])
# #     env = FlattenObservationWrapper(env)
# #     env = LogWrapper(env)
# #
# #     # We assume the environment has a default_params dict, just like in TabularEnv.
# #     env_params = env.default_params
# #
# #     # Calculate how many updates and minibatch sizes
# #     config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
# #     config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]
# #
# #     # Learning rate schedule
# #     def linear_schedule(count):
# #         frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
# #         return config["LR"] * frac
# #
# #     def train(rng):
# #         # ----------------------
# #         # 2. Init Policy Network
# #         # ----------------------
# #         network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
# #         rng, init_rng = jax.random.split(rng)
# #         init_obs = jnp.zeros(env.observation_space(env_params).shape)
# #         network_params = network.init(init_rng, init_obs)
# #
# #         # ----------------------
# #         # 3. Define Optimizer
# #         # ----------------------
# #         if config["ANNEAL_LR"]:
# #             tx = optax.chain(
# #                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
# #                 optax.adam(learning_rate=linear_schedule, eps=1e-5)
# #             )
# #         else:
# #             tx = optax.chain(
# #                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
# #                 optax.adam(config["LR"], eps=1e-5)
# #             )
# #         train_state = TrainState.create(
# #             apply_fn=network.apply,
# #             params=network_params,
# #             tx=tx,
# #         )
# #
# #         # ----------------------
# #         # 4. Initialize Env
# #         # ----------------------
# #         rng, reset_rng = jax.random.split(rng)
# #         reset_keys = jax.random.split(reset_rng, config["NUM_ENVS"])
# #         obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
# #
# #         # ----------------------
# #         # 5. Training Loop
# #         # ----------------------
# #         def _update_step(runner_state, _):
# #             """
# #             Runs one PPO update, collecting trajectories via rollouts,
# #             then updating the network.
# #             """
# #             # --------------------------------
# #             # 5a. Collect Trajectories (Rollout)
# #             # --------------------------------
# #             def _env_step(runner_state, _):
# #                 train_state, env_state, last_obs, rng_step = runner_state
# #
# #                 # Select action
# #                 rng_step, act_rng = jax.random.split(rng_step)
# #                 pi, value = network.apply(train_state.params, last_obs)
# #                 action = pi.sample(seed=act_rng)
# #                 log_prob = pi.log_prob(action)
# #
# #                 # Step environment
# #                 rng_step, step_rng = jax.random.split(rng_step)
# #                 step_keys = jax.random.split(step_rng, config["NUM_ENVS"])
# #                 obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
# #                     step_keys, env_state, action, env_params
# #                 )
# #
# #                 transition = Transition(done, action, value, reward, log_prob, last_obs, info)
# #                 next_runner_state = (train_state, env_state, obsv, rng_step)
# #                 return next_runner_state, transition
# #
# #             runner_state, traj_batch = jax.lax.scan(
# #                 _env_step, runner_state, None, config["NUM_STEPS"]
# #             )
# #
# #             # --------------------------------
# #             # 5b. Calculate Advantages (GAE)
# #             # --------------------------------
# #             train_state, env_state, last_obs, rng_step = runner_state
# #             _, last_val = network.apply(train_state.params, last_obs)
# #
# #             def _calculate_gae(traj_batch, last_val):
# #                 def _get_advantages(gae_and_next_value, transition):
# #                     gae, next_value = gae_and_next_value
# #                     done, value, reward = transition.done, transition.value, transition.reward
# #
# #                     delta = reward + config["GAMMA"] * next_value * (1 - done) - value
# #                     gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
# #                     return (gae, value), gae
# #
# #                 # Reverse scan to compute advantage
# #                 init_gae = (jnp.zeros_like(last_val), last_val)
# #                 _, advantages = jax.lax.scan(
# #                     _get_advantages, init_gae, traj_batch, reverse=True, unroll=16
# #                 )
# #                 return advantages, advantages + traj_batch.value
# #
# #             advantages, targets = _calculate_gae(traj_batch, last_val)
# #
# #             # --------------------------------
# #             # 5c. Update the Network via PPO
# #             # --------------------------------
# #             def _update_epoch(update_state, _):
# #                 def _update_minbatch(train_state, batch_data):
# #                     traj_batch_part, advantages_part, targets_part = batch_data
# #
# #                     def _loss_fn(params, tbatch, gae, tgts):
# #                         pi, value = network.apply(params, tbatch.obs)
# #                         log_prob = pi.log_prob(tbatch.action)
# #
# #                         # Value loss (clipped)
# #                         value_pred_clipped = tbatch.value + (value - tbatch.value).clip(
# #                             -config["CLIP_EPS"], config["CLIP_EPS"]
# #                         )
# #                         val_loss1 = jnp.square(value - tgts)
# #                         val_loss2 = jnp.square(value_pred_clipped - tgts)
# #                         value_loss = 0.5 * jnp.maximum(val_loss1, val_loss2).mean()
# #
# #                         # Actor loss (clipped)
# #                         ratio = jnp.exp(log_prob - tbatch.log_prob)
# #                         gae_std = (gae - gae.mean()) / (gae.std() + 1e-8)
# #                         loss_actor1 = ratio * gae_std
# #                         loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_std
# #                         actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
# #
# #                         # Entropy bonus
# #                         entropy = pi.entropy().mean()
# #
# #                         total_loss = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
# #                         return total_loss, (value_loss, actor_loss, entropy)
# #
# #                     grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
# #                     (loss_val, (v_loss, a_loss, entropy)), grads = grad_fn(
# #                         train_state.params, tbatch, advantages_part, targets_part
# #                     )
# #                     train_state = train_state.apply_gradients(grads=grads)
# #
# #                     # (Optional) Could log more details here if desired
# #                     return train_state, (loss_val, v_loss, a_loss, entropy)
# #
# #                 train_state, traj_batch_all, advantages_all, targets_all, rng_step = update_state
# #                 rng_step, perm_rng = jax.random.split(rng_step)
# #
# #                 # Shuffle and create minibatches
# #                 batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
# #                 assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], \
# #                     "Batch size must match total number of transitions (NUM_STEPS * NUM_ENVS)"
# #
# #                 permutation = jax.random.permutation(perm_rng, batch_size)
# #                 full_batch = (traj_batch_all, advantages_all, targets_all)
# #
# #                 # Flatten across steps & envs
# #                 full_batch = jax.tree_util.tree_map(
# #                     lambda x: x.reshape((batch_size,) + x.shape[2:]), full_batch
# #                 )
# #
# #                 # Shuffle
# #                 shuffled_batch = jax.tree_util.tree_map(
# #                     lambda x: jnp.take(x, permutation, axis=0), full_batch
# #                 )
# #
# #                 # Create minibatches
# #                 minibatches = jax.tree_util.tree_map(
# #                     lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
# #                     shuffled_batch,
# #                 )
# #
# #                 train_state, losses_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
# #                 update_state = (train_state, traj_batch_all, advantages_all, targets_all, rng_step)
# #                 return update_state, losses_info
# #
# #             # Scan over multiple epochs
# #             update_state = (train_state, traj_batch, advantages, targets, rng_step)
# #             update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
# #             train_state = update_state[0]
# #             final_runner_state = (train_state, env_state, last_obs, rng_step)
# #
# #             # For logging, let's store info from transitions
# #             metric = traj_batch.info
# #             return final_runner_state, metric
# #
# #         # Initialize scan state
# #         rng, _rng = jax.random.split(rng)
# #         runner_state = (train_state, env_state, obsv, _rng)
# #
# #         # Run the outer training loop
# #         runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
# #         return {"runner_state": runner_state, "metrics": metric}
# #
# #     return train
# #
# #
# # # ------------------------------------------------------------------------------
# # # Minimal "main" script example: Run training + log with W&B + local plotting
# # # ------------------------------------------------------------------------------
# # if __name__ == "__main__":
# #     import time
# #     import matplotlib.pyplot as plt
# #
# #     # Example config
# #     config = {
# #         "PROBLEM_FILE": "minigrid/consolidated.npz",  # <--- Your custom environment file
# #         "LR": 2.5e-4,
# #         "NUM_ENVS": 4,
# #         "NUM_STEPS": 128,
# #         "TOTAL_TIMESTEPS": 5e5,
# #         "UPDATE_EPOCHS": 4,
# #         "NUM_MINIBATCHES": 4,
# #         "GAMMA": 0.99,
# #         "GAE_LAMBDA": 0.95,
# #         "CLIP_EPS": 0.2,
# #         "ENT_COEF": 0.01,
# #         "VF_COEF": 0.5,
# #         "MAX_GRAD_NORM": 0.5,
# #         "ACTIVATION": "tanh",
# #         "ANNEAL_LR": True,
# #     }
# #
# #     # Initialize W&B
# #     wandb.init(project="MyTabularPPOProject", config=config)
# #
# #     rng = jax.random.PRNGKey(42)
# #     train_fn = make_train(config)
# #     train_jit = jax.jit(train_fn)
# #
# #     t0 = time.time()
# #     out = jax.block_until_ready(train_jit(rng))
# #     elapsed = time.time() - t0
# #     print(f"Training time: {elapsed:.2f} s")
# #
# #     # Example: If your environment logs "returned_episode_returns" or something similar, you can log & plot it:
# #     # (In your custom environment, you could accumulate episode returns in the 'info' dict if you want.)
# #     if "returned_episode_returns" in out["metrics"]:
# #         # Suppose you stored episodic returns across each environment. E.g. shape: [NUM_STEPS, NUM_ENVS]
# #         avg_returns = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
# #         for step_i, val in enumerate(avg_returns):
# #             wandb.log({"AverageReturn": float(val)}, step=step_i)
# #
# #         # Local plot
# #         plt.plot(avg_returns)
# #         plt.xlabel("Update Step")
# #         plt.ylabel("Return")
# #         plt.title("PPO on Custom TabularEnv")
# #         plt.show()
# #     else:
# #         print("No 'returned_episode_returns' found in metrics. Modify your environment logging if needed.")
