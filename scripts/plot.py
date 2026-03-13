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
    plt.plot(timesteps, mean_rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation Reward")
    plt.show()


if __name__ == "__main__":
    main()