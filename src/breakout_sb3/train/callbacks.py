from pathlib import Path
import csv
import numpy as np

from stable_baselines3.common.callbacks import EvalCallback


class EvalAndCheckpointCallback(EvalCallback):
    """
    1. Save best model according to mean reward
    2. Save one checkpoint at every evaluation point
    3. Save evaluation statistics to a CSV file
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: Path,
        log_path: Path,
        checkpoint_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.eval_csv_path = Path(log_path) / "eval_metrics.csv"
        Path(log_path).mkdir(parents=True, exist_ok=True)

        super().__init__(
            eval_env=eval_env,
            best_model_save_path=str(best_model_save_path),
            log_path=str(log_path),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )

        self._init_csv()

    def _init_csv(self):
        if not self.eval_csv_path.exists():
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timesteps",
                        "mean",
                        "std",
                        "min",
                        "p10",
                        "p50",
                        "p90",
                        "max",
                        "checkpoint_path",
                    ]
                )

    def _save_eval_row(self, timestep: int, rewards: np.ndarray, checkpoint_path: str):
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        min_reward = float(np.min(rewards))
        p10_reward = float(np.percentile(rewards, 10))
        p50_reward = float(np.percentile(rewards, 50))
        p90_reward = float(np.percentile(rewards, 90))
        max_reward = float(np.max(rewards))

        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestep,
                    mean_reward,
                    std_reward,
                    min_reward,
                    p10_reward,
                    p50_reward,
                    p90_reward,
                    max_reward,
                    checkpoint_path,
                ]
            )

    def _on_step(self) -> bool:
        prev_eval_count = (
            len(self.evaluations_timesteps)
            if self.evaluations_timesteps is not None
            else 0
        )

        continue_training = super()._on_step()

        new_eval_count = (
            len(self.evaluations_timesteps)
            if self.evaluations_timesteps is not None
            else 0
        )

        # If a new evaluation happened, save checkpoint + metrics
        if new_eval_count > prev_eval_count:
            timestep = int(self.num_timesteps)

            checkpoint_path = self.checkpoint_dir / f"ckpt_{timestep}.zip"
            self.model.save(str(checkpoint_path))

            rewards = np.array(self.evaluations_results[-1], dtype=np.float32)
            self._save_eval_row(
                timestep=timestep,
                rewards=rewards,
                checkpoint_path=str(checkpoint_path),
            )

            if self.verbose > 0:
                mean_reward = float(np.mean(rewards))
                p10_reward = float(np.percentile(rewards, 10))
                p90_reward = float(np.percentile(rewards, 90))
                print(
                    f"[Checkpoint Saved] step={timestep} "
                    f"mean={mean_reward:.2f} "
                    f"p10={p10_reward:.2f} "
                    f"p90={p90_reward:.2f} "
                    f"path={checkpoint_path}"
                )

        return continue_training


def build_callbacks(cfg: dict, eval_env, run_dir: Path):
    eval_log_dir = run_dir / "eval_logs"
    best_model_dir = run_dir / "best_model"
    checkpoint_dir = run_dir / "checkpoints"

    eval_log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_freq = cfg["train"].get("eval_freq", 50_000)
    n_eval_episodes = cfg["train"].get("n_eval_episodes", 5)

    eval_callback = EvalAndCheckpointCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        checkpoint_dir=checkpoint_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    return eval_callback