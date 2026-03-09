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