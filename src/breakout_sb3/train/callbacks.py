from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path


def build_callbacks(cfg: dict, eval_env, run_dir: Path):

    eval_log_dir = run_dir / "eval_logs"
    best_model_dir = run_dir / "best_model"

    eval_log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    eval_freq = cfg["train"].get("eval_freq", 50_000)
    n_eval_episodes = cfg["train"].get("n_eval_episodes", 5)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    return eval_callback