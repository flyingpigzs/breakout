from datetime import datetime
from pathlib import Path
import json
import torch

from stable_baselines3 import DQN, PPO

from breakout_sb3.envs.make_env import make_eval_env, make_train_env
from breakout_sb3.train.callbacks import build_callbacks


def _get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"


def _create_run_dir(cfg: dict) -> Path:
    base_dir = Path("outputs") / "runs"
    base_dir.mkdir(parents=True, exist_ok=True)

    algo_name = cfg["algo"].get("name", "unknown").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_dir / f"{timestamp}_breakout_{algo_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def _save_config_snapshot(cfg: dict, run_dir: Path):
    snapshot_path = run_dir / "config_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _build_model(cfg: dict, train_env, run_dir: Path):
    algo_cfg = cfg["algo"]
    train_cfg = cfg["train"]

    algo_name = algo_cfg["name"].lower()

    device = _get_device()

    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    resume_path = train_cfg.get("resume_model", None)

    if resume_path is not None:

        print("Resuming from:", resume_path)

        if algo_name == "ppo":
            model = PPO.load(
                resume_path,
                env=train_env,
                device=device,
                tensorboard_log=str(tb_dir),
            )

        elif algo_name == "dqn":
            model = DQN.load(
                resume_path,
                env=train_env,
                device=device,
                tensorboard_log=str(tb_dir),
            )

        else:
            raise ValueError("Unknown algo")

        return model

    if algo_name == "dqn":
        model = DQN(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=algo_cfg.get("learning_rate", 1e-4),
            buffer_size=algo_cfg.get("buffer_size", 100_000),
            learning_starts=algo_cfg.get("learning_starts", 50_000),
            batch_size=algo_cfg.get("batch_size", 32),
            gamma=algo_cfg.get("gamma", 0.99),
            train_freq=algo_cfg.get("train_freq", 4),
            gradient_steps=algo_cfg.get("gradient_steps", 1),
            target_update_interval=algo_cfg.get("target_update_interval", 10_000),
            exploration_fraction=algo_cfg.get("exploration_fraction", 0.1),
            exploration_initial_eps=algo_cfg.get("exploration_initial_eps", 1.0),
            exploration_final_eps=algo_cfg.get("exploration_final_eps", 0.01),
            max_grad_norm=algo_cfg.get("max_grad_norm", 10.0),
            tensorboard_log=str(tb_dir),
            verbose=1,
            device=device,
        )
        return model

    if algo_name == "ppo":
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=algo_cfg.get("learning_rate", 2.5e-4),
            n_steps=algo_cfg.get("n_steps", 128),
            batch_size=algo_cfg.get("batch_size", 256),
            n_epochs=algo_cfg.get("n_epochs", 4),
            gamma=algo_cfg.get("gamma", 0.99),
            gae_lambda=algo_cfg.get("gae_lambda", 0.95),
            clip_range=algo_cfg.get("clip_range", 0.1),
            ent_coef=algo_cfg.get("ent_coef", 0.01),
            vf_coef=algo_cfg.get("vf_coef", 0.5),
            max_grad_norm=algo_cfg.get("max_grad_norm", 0.5),
            tensorboard_log=str(tb_dir),
            verbose=1,
            device=device,
        )
        return model

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def train(cfg: dict):
    train_cfg = cfg["train"]
    total_timesteps = train_cfg.get("total_timesteps", 1_000_000)

    run_dir = _create_run_dir(cfg)
    _save_config_snapshot(cfg, run_dir)

    train_env = make_train_env(cfg)
    eval_env = make_eval_env(cfg)

    callback = build_callbacks(cfg, eval_env=eval_env, run_dir=run_dir)
    model = _build_model(cfg, train_env=train_env, run_dir=run_dir)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))

    train_env.close()
    eval_env.close()

    return run_dir