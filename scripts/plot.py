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

    p10_rewards = np.percentile(results, 10, axis=1)
    p50_rewards = np.percentile(results, 50, axis=1)
    p90_rewards = np.percentile(results, 90, axis=1)

    # ---- print statistics ----
    for t, mean_r, std_r, min_r, p10_r, p50_r, p90_r, max_r in zip(
        timesteps,
        mean_rewards,
        std_rewards,
        min_rewards,
        p10_rewards,
        p50_rewards,
        p90_rewards,
        max_rewards,
    ):
        print(
            f"t={t:>9d}  "
            f"mean={mean_r:7.2f}  "
            f"std={std_r:7.2f}  "
            f"min={min_r:7.2f}  "
            f"p10={p10_r:7.2f}  "
            f"p50={p50_r:7.2f}  "
            f"p90={p90_r:7.2f}  "
            f"max={max_r:7.2f}"
        )

    # ---- plot ----
    plt.figure(figsize=(12, 7))

    plt.plot(timesteps, mean_rewards, label="mean")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="mean ± std",
    )

    plt.plot(timesteps, p10_rewards, linestyle="--", label="p10")
    plt.plot(timesteps, p50_rewards, linestyle="-.", label="p50 (median)")
    plt.plot(timesteps, p90_rewards, linestyle="--", label="p90")

    # Optional: keep min/max for reference
    plt.plot(timesteps, min_rewards, linestyle=":", label="min")
    plt.plot(timesteps, max_rewards, linestyle=":", label="max")

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Evaluation Reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()