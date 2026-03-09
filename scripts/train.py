from breakout_sb3.train.train import train
from breakout_sb3.utils.config import load_config


def main():

    cfg = load_config(
        "configs/env.yaml",
        "configs/algo_dqn.yaml",
        "configs/train.yaml",
    )

    train(cfg)


if __name__ == "__main__":
    main()