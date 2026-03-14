import gymnasium as gym
import numpy as np

class ClipRewardEnv(gym.RewardWrapper):

    def reward(self, reward):
        return np.sign(reward)

class CropObservation(gym.ObservationWrapper):

    def __init__(self, env, top=34, bottom=194):
        super().__init__(env)
        self.top = top
        self.bottom = bottom

        obs_shape = env.observation_space.shape
        new_shape = (bottom - top, obs_shape[1], obs_shape[2])

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        return obs[self.top:self.bottom]

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life look like end-of-episode to the agent,
    but only reset the environment when the game is truly over.

    This is the classic Atari trick used in DeepMind / Baselines / SB3:
    - losing a life -> terminated=True (for training)
    - but underlying env is NOT reset
    - only real game over triggers true reset
    """

    def __init__(self, env):
        super().__init__(env)

        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # True game over from env
        self.was_real_done = terminated or truncated

        # Get current lives from ALE
        lives = self.env.unwrapped.ale.lives()

        # Detect life loss (but not game over)
        if lives < self.lives and lives > 0:
            terminated = True

        self.lives = lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset only when real game over.
        If life was lost but game not over, continue from same state.
        """

        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Continue from same state after life loss
            obs, reward, terminated, truncated, info = self.env.step(0)

            # Rare case: step leads to game over
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        self.lives = self.env.unwrapped.ale.lives()

        return obs, info