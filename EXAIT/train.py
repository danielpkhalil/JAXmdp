import argparse
import json
from ppo import run_ppo
from dqn import run_dqn

# default configurations
DEFAULT_PPO_CONFIG = {
    "SEED": 1,
    "LR": 2.5e-4,
    "NUM_ENVS": 8,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.1,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ENV_NAME": "TabularMDP",
    "ENV_FILE": "test.npz",
    "REWARD_SCALE": 1.0/100,
    "EVAL_FREQUENCY": 1000,
    "TRAIN_MEDIAN_WINDOW": 20,
    "OPTIMAL_REWARD": 1.0,
}

DEFAULT_DQN_CONFIG = {
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
    "REWARD_SCALE": 1.0/100,
    "SEED": 0,
    "NUM_SEEDS": 1,
    "WANDB_MODE": "online",
    # New eval/stop params
    "EVAL_FREQUENCY": 1000,
    "TRAIN_MEDIAN_WINDOW": 20,
    "OPTIMAL_REWARD": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO or DQN on Gymnax environments")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo", help="which algorithm to run")
    parser.add_argument("--config", type=str, help="path to a JSON config file")
    parser.add_argument("--env-file", type=str, help="path to npz ENV file (for TabularMDP)")
    return parser.parse_args()


def main():
    args = parse_args()
    # load or select default config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_PPO_CONFIG.copy() if args.algo == "ppo" else DEFAULT_DQN_CONFIG.copy()

    # override env file if provided
    if args.env_file:
        config["ENV_FILE"] = args.env_file

    # dispatch
    if args.algo == "ppo":
        run_ppo(config)
    else:
        run_dqn(config)

if __name__ == "__main__":
    main()