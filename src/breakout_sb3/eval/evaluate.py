from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from breakout_sb3.envs.make_env import make_eval_env


def evaluate_model(
    cfg: dict,
    model_path: str,
    n_eval_episodes: int = 5,
    render_mode: str | None = None,
):
    """
    Evaluate a trained DQN model on the evaluation environment.
    """
    env = make_eval_env(cfg, render_mode=render_mode)

    model = DQN.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    env.close()

    return mean_reward, std_reward