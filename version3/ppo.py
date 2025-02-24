import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from jax.experimental import host_callback as hcb
import wandb

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
        # If the input is an image, normalize and flatten.
        if x.ndim > 2:
            x = x.astype(jnp.float32) / 255.0
            x = x.reshape((x.shape[0], -1))
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Actor network.
        actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_hidden = activation_fn(actor_hidden)
        actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_hidden)
        actor_hidden = activation_fn(actor_hidden)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_hidden)
        pi = distrax.Categorical(logits=logits)

        # Critic network.
        critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic_hidden = activation_fn(critic_hidden)
        critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic_hidden)
        critic_hidden = activation_fn(critic_hidden)
        critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_hidden)
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
# One update step (jitted) that rolls out trajectories and applies PPO updates.
# ------------------------------------------------------------------------------
def update_step(runner_state, config, network, env, env_params):
    train_state, env_state, last_obs, rng = runner_state

    def env_step(runner_state, _):
        train_state, env_state, last_obs, rng = runner_state
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        transition = Transition(done, action, value, reward, log_prob, last_obs, info)
        new_state = (train_state, env_state, obsv, rng)
        return new_state, transition

    runner_state, traj_batch = jax.lax.scan(env_step, runner_state, None, config["NUM_STEPS"])
    train_state, env_state, last_obs, rng = runner_state
    _, last_val = network.apply(train_state.params, last_obs)

    # Compute advantages via Generalized Advantage Estimation.
    def calc_gae(traj_batch, last_val):
        def gae_scan(carry, transition):
            gae, next_value = carry
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(gae_scan, (jnp.zeros_like(last_val), last_val),
                                     traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value

    advantages, targets = calc_gae(traj_batch, last_val)

    # PPO update epochs over minibatches.
    def update_epoch(update_state, _):
        def update_minibatch(train_state, batch_info):
            traj_batch, gae, targets = batch_info
            def loss_fn(params, traj_batch, gae, targets):
                pi, value = network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                v_loss1 = jnp.square(value - targets)
                v_loss2 = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(ratio * gae_norm,
                                          jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm).mean()
                entropy = pi.entropy().mean()
                total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                return total_loss, (value_loss, loss_actor, entropy)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (total_loss, aux_vals), grads = grad_fn(train_state.params, traj_batch, gae, targets)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        train_state, traj_batch, gae, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        permutation = jax.random.permutation(_rng, batch_size)
        batch = (traj_batch, gae, targets)
        batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
        shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
            shuffled_batch
        )
        train_state, total_loss = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (train_state, traj_batch, gae, targets, rng), total_loss

    update_state = (train_state, traj_batch, advantages, targets, rng)
    update_state, loss_info = jax.lax.scan(update_epoch, update_state, None, config["UPDATE_EPOCHS"])
    train_state = update_state[0]
    metric = traj_batch.info  # You may extend this to include more metrics.
    new_runner_state = (train_state, env_state, last_obs, rng)
    return new_runner_state, metric

# ------------------------------------------------------------------------------
# Evaluation function (deterministic policy)
# ------------------------------------------------------------------------------
def evaluate_policy(train_state, env, env_params, network, num_envs, obs_shape):
    rng = jax.random.PRNGKey(0)
    reset_rng = jax.random.split(rng, num_envs)
    obs, env_state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_rng, env_params)
    done = jnp.zeros(num_envs, dtype=bool)
    total_reward = 0.0
    steps = 0
    while not jnp.all(done):
        pi, _ = network.apply(train_state.params, obs)
        # Deterministic: choose action with highest probability.
        action = jnp.argmax(pi.probs, axis=-1)
        rng, _ = jax.random.split(rng)
        rng_step = jax.random.split(rng, num_envs)
        obs, env_state, reward, done, info = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
            rng_step, env_state, action, env_params
        )
        total_reward += reward.sum()
        steps += num_envs
    return float(total_reward), steps

# ------------------------------------------------------------------------------
# Main training loop (unrolled in Python to allow evaluation every eval_freq steps)
# ------------------------------------------------------------------------------
def train_loop(config, rng, network, env, env_params, obs_shape):
    # Initialize network parameters, optimizer, and environment.
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((config["NUM_ENVS"],) + obs_shape, dtype=jnp.uint8)
    network_params = network.init(_rng, init_x)
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lambda count: config["LR"] * (1 - count / config["NUM_UPDATES"]), eps=1e-5)
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5)
        )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_rng, env_params)
    runner_state = (train_state, env_state, obsv, rng)

    global_steps = 0
    T_PPO_SB3 = None  # Step when optimal reward first reached.
    eval_results = []

    # JIT the update step.
    update_step_jit = jax.jit(lambda rs: update_step(rs, config, network, env, env_params))

    for update in range(config["NUM_UPDATES"]):
        runner_state, metric = update_step_jit(runner_state)
        global_steps += config["NUM_STEPS"]
        # Evaluate every eval_freq timesteps.
        if global_steps % config["eval_freq"] < config["NUM_STEPS"]:
            eval_reward, eval_steps = evaluate_policy(runner_state[0], env, env_params, network, config["NUM_ENVS"], obs_shape)
            wandb.log({"eval_reward": eval_reward, "global_steps": global_steps})
            eval_results.append((global_steps, eval_reward))
            if eval_reward >= config["J_opt"] and T_PPO_SB3 is None:
                T_PPO_SB3 = global_steps
                wandb.run.summary["T_PPO_SB3"] = T_PPO_SB3
    # Final evaluation.
    final_eval_reward, _ = evaluate_policy(runner_state[0], env, env_params, network, config["NUM_ENVS"], obs_shape)
    wandb.log({"final_eval_reward": final_eval_reward, "global_steps": global_steps})
    wandb.run.summary["J_PPO_SB3"] = final_eval_reward
    return eval_results

# ------------------------------------------------------------------------------
# Main entry point with WandB initialization
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Base configuration.
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,            # Adjust as needed (e.g., 4)
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 1,      # Adjust as needed (e.g., 4)
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        # Evaluation settings.
        "eval_freq": 1000,         # Evaluate every 1000 timesteps.
        "J_opt": 200.0,            # Optimal reward threshold.
    }
    # Compute derived configuration values.
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    # Initialize wandb.
    wandb.init(project="MyTabularPPOProject", config=config)

    # Create environment and network.
    env = create_tabular_env("consolidated.npz")
    env = LogWrapper(env)
    env_params = env.default_params
    obs_shape = env.observation_space(env_params).shape
    network = ActorCritic(action_dim=env.action_space(env_params).n, activation=config["ACTIVATION"])

    rng = jax.random.PRNGKey(30)
    eval_results = train_loop(config, rng, network, env, env_params, obs_shape)
    print("Training finished.")
    print("Evaluation results (global_steps, eval_reward):")
    print(eval_results)
