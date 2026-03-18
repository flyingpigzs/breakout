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


class FireResetEnv(gym.Wrapper):
    """
    Automatically press FIRE:
    - after a true reset
    - after life loss (when the game waits for a new launch)

    Assumes action meanings include "FIRE".
    """

    def __init__(self, env):
        super().__init__(env)

        if not hasattr(env.unwrapped, "get_action_meanings"):
            raise ValueError("Environment does not provide action meanings().")

        self.action_meanings = env.unwrapped.get_action_meanings()
        if "FIRE" not in self.action_meanings:
            raise ValueError("Environment does not support FIRE action.")

        self.fire_action = self.action_meanings.index("FIRE")
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()

        # Start the game
        obs, reward, terminated, truncated, info = self.env.step(self.fire_action)

        # Rare safeguard
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            self.lives = self.env.unwrapped.ale.lives()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        new_lives = self.env.unwrapped.ale.lives()
        life_lost = (new_lives < self.lives) and (new_lives > 0)

        if life_lost and not (terminated or truncated):
            # Relaunch ball automatically after losing a life
            obs, extra_reward, terminated, truncated, info = self.env.step(self.fire_action)
            reward += extra_reward

        self.lives = new_lives
        return obs, reward, terminated, truncated, info