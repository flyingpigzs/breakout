import numpy as np
import matplotlib.pyplot as plt

data = np.load("outputs/runs/20260310_214724_breakout_dqn/eval_logs/evaluations.npz")

timesteps = data["timesteps"]
results = data["results"]

mean_rewards = results.mean(axis=1)
plt.plot(timesteps, mean_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Evaluation Reward")
plt.show()