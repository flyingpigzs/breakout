import argparse

from breakout_sb3.eval.evaluate import evaluate_model
from breakout_sb3.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
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

    mean_reward, std_reward = evaluate_model(
        cfg=cfg,
        model_path=args.model_path,
        n_eval_episodes=args.episodes,
        render_mode="human",
    )

    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")


if __name__ == "__main__":
    main()