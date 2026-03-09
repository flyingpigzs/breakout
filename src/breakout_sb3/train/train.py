from pathlib import Path
from datetime import datetime

from stable_baselines3 import DQN

from breakout_sb3.envs.make_env import make_train_env, make_eval_env
from breakout_sb3.train.callbacks import build_callbacks


def train(cfg: dict):

    total_timesteps = cfg["train"].get("total_timesteps", 1_000_000)

    # create run directory
    run_dir = Path("outputs/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # create environments
    train_env = make_train_env(cfg)
    eval_env = make_eval_env(cfg)

    # build callback
    callback = build_callbacks(cfg, eval_env, run_dir)

    # create model
    model = DQN(
        "CnnPolicy",
        train_env,
        learning_rate=cfg["algo"].get("learning_rate", 1e-4),
        buffer_size=cfg["algo"].get("buffer_size", 100000),
        learning_starts=cfg["algo"].get("learning_starts", 50000),
        batch_size=cfg["algo"].get("batch_size", 32),
        gamma=cfg["algo"].get("gamma", 0.99),
        train_freq=cfg["algo"].get("train_freq", 4),
        target_update_interval=cfg["algo"].get("target_update_interval", 10000),
        exploration_fraction=cfg["algo"].get("exploration_fraction", 0.1),
        exploration_final_eps=cfg["algo"].get("exploration_final_eps", 0.01),
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    model.save(run_dir / "final_model")

    train_env.close()
    eval_env.close()