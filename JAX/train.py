import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from .algorithms.dqn import run_dqn
from .algorithms.ppo import run_ppo_training
from .gymnax_env import TabularEnv

ex = Experiment("train_sb3")


@ex.config
def sacred_config(_log):
    env_file = "./JAX/test.npz"
    env_name = os.path.splitext(os.path.basename(env_file))[0]
    horizon = 10  # noqa: F841
    reward_scale = 1.0  # noqa: F841
    use_action_mask = True # noqa: F841

    alg = "DQN"  # or "DQN"  # noqa: F841
    total_timesteps = 1e6
    gamma = 0.99
    seed = 0  # noqa: F841

    eval_frequency = 1000  # noqa: F841
    stop_on_eval_reward = np.inf  # noqa: F841
    stop_on_median_train_reward = np.inf  # noqa: F841

    if alg == "PPO":
        algo_args = {  # noqa: F841
            "lr": 2.5e-4,
            "num_envs": 8,
            "num_steps": 128,
            "update_epochs": 4,
            "minibatch_size": 256,
            "gamma": gamma,
            "gae_lambda": 0.95,
            "clip_eps": 0.1,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "relu",
        }
    elif alg == "DQN":
        algo_args = {  # noqa: F841
            "num_envs": 10,
            "buffer_size": 10000,
            "buffer_batch_size": 128,
            "epsilon_start": 1.0,
            "epsilon_finish": 0.05,
            "epsilon_anneal_time": total_timesteps,
            "target_update_interval": 1000,
            "lr": 1e-4,
            "learning_starts": 10_000,
            "training_interval": 10,
            "lr_linear_decay": False,
            "tau": 1.0,
            "gamma": gamma,
        }

    log_dir = "data/logs"
    experiment_tag = None
    experiment_name_parts = [alg, env_name]
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name_parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    experiment_dir = os.path.join(log_dir, *experiment_name_parts)
    observer = FileStorageObserver(experiment_dir)
    ex.observers.append(observer)


@ex.automain
def main(
    env_file,
    horizon,
    reward_scale,
    use_action_mask,
    alg,
    total_timesteps,
    seed,
    eval_frequency,
    stop_on_eval_reward,
    stop_on_median_train_reward,
    algo_args,
    experiment_dir,
    _log,
):
    env = TabularEnv(env_file)
    env_params = env.default_params().replace(
        reward_scale=reward_scale,
        horizon=horizon,
        use_action_mask=use_action_mask,
    )
    if alg == "PPO":
        run_ppo_training(
            env=env,
            env_params=env_params,
            total_timesteps=total_timesteps,
            seed=seed,
            eval_frequency=eval_frequency,
            stop_on_eval_reward=stop_on_eval_reward,
            stop_on_median_train_reward=stop_on_median_train_reward,
            algo_args=algo_args,
            log_dir=experiment_dir,
        )
    else:
        run_dqn(
            env=env,
            env_params=env_params,
            total_timesteps=total_timesteps,
            seed=seed,
            eval_frequency=eval_frequency,
            stop_on_eval_reward=stop_on_eval_reward,
            stop_on_median_train_reward=stop_on_median_train_reward,
            algo_args=algo_args,
            log_dir=experiment_dir,
        )


# if __name__ == "__main__":
#     main()
