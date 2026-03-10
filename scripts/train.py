import argparse

from breakout_sb3.train.train import train
from breakout_sb3.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    args = parser.parse_args()

    if args.algo == "dqn":
        algo_config_path = "configs/algo_dqn.yaml"
    else:
        algo_config_path = "configs/algo_ppo.yaml"

    cfg = load_config(
        "configs/env.yaml",
        algo_config_path,
        "configs/train.yaml",
    )

    run_dir = train(cfg)
    print(f"Training finished. Run directory: {run_dir}")


if __name__ == "__main__":
    main()