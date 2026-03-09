from breakout_sb3.eval.evaluate import evaluate_model
from breakout_sb3.utils.config import load_config


def main():
    cfg = load_config(
        "configs/env.yaml",
        "configs/algo_dqn.yaml",
        "configs/train.yaml",
    )

    # model_path = "outputs/runs/20260307_211503/final_model.zip"
    model_path = "outputs/runs/20260308_112312/best_model/best_model.zip"

    mean_reward, std_reward = evaluate_model(
        cfg=cfg,
        model_path=model_path,
        n_eval_episodes=5,
        render_mode="human",
    )

    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")


if __name__ == "__main__":
    main()