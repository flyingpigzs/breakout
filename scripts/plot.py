import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()

    data = np.load(args.log_path)

    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    min_rewards = results.min(axis=1)
    max_rewards = results.max(axis=1)

    # ---- print statistics ----
    for t, mean_r, std_r, min_r, max_r in zip(
        timesteps,
        mean_rewards,
        std_rewards,
        min_rewards,
        max_rewards,
    ):
        print(
            f"t={t:>9d}  "
            f"mean={mean_r:6.2f}  "
            f"std={std_r:6.2f}  "
            f"min={min_r:6.2f}  "
            f"max={max_r:6.2f}"
        )

    # ---- plot ----
    plt.figure(figsize=(10, 6))

    plt.plot(timesteps, mean_rewards, label="mean")

    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="mean ± std",
    )

    plt.plot(timesteps, min_rewards, linestyle="--", label="min")
    plt.plot(timesteps, max_rewards, linestyle="--", label="max")

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Evaluation Reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()