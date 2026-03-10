from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from breakout_sb3.envs.make_env import make_eval_env


def _load_model(cfg: dict, model_path: str):
    algo_name = cfg["algo"]["name"].lower()

    if algo_name == "dqn":
        return DQN.load(model_path)
    if algo_name == "ppo":
        return PPO.load(model_path)

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def evaluate_model(
    cfg: dict,
    model_path: str,
    n_eval_episodes: int = 5,
    render_mode: str | None = None,
):
    env = make_eval_env(cfg, render_mode=render_mode)
    model = _load_model(cfg, model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    env.close()
    return mean_reward, std_reward