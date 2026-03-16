import ale_py
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv, VecTransposeImage

from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

from breakout_sb3.envs.wrappers import ClipRewardEnv, CropObservation, EpisodicLifeEnv, FireResetEnv

# Create the raw Gymnasium env instance. No wrappers here.
def _make_base_env(env_id: str, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode)
    return env

# Apply observation/reward/action wrappers that should be identical across training and evaluation
def _apply_wrappers(env, cfg, is_eval):
    resize_shape = cfg["env"].get("resize_shape", [84, 84])
    env = Monitor(env)
    env = FireResetEnv(env)

    if not is_eval:
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)

    env = CropObservation(env)
    env = ResizeObservation(env, shape=tuple(resize_shape))
    env = GrayscaleObservation(env, keep_dim=True)

    return env

def make_train_env(cfg: dict):
    env_id = cfg["env"]["id"]
    n_envs = cfg["env"].get("n_envs", 4)
    frame_stack = cfg["env"].get("frame_stack", 4)

    def make_env_fn():
        def _init():
            env = _make_base_env(
                env_id=env_id,
                render_mode=None,
            )
            env = _apply_wrappers(env, cfg, is_eval=False)
            return env

        return _init

    env_fns = [make_env_fn() for _ in range(n_envs)]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    vec_env = VecTransposeImage(vec_env)

    return vec_env

def make_eval_env(cfg: dict, render_mode: str | None = None):
    env_id = cfg["env"]["id"]
    frame_stack = cfg["env"].get("frame_stack", 4)

    def make_env_fn():
        def _init():
            env = _make_base_env(
                env_id=env_id,
                render_mode=render_mode,
            )
            env = _apply_wrappers(env, cfg, is_eval=True)
            return env

        return _init

    env_fns = [make_env_fn()]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    vec_env = VecTransposeImage(vec_env)

    return vec_env
