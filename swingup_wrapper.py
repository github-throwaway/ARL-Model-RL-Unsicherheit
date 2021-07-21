# coding: utf-8
import math
from random import uniform

import gym
import gym_cartpole_swingup
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class SwingUpWrapper():
    # Could be one of:
    # CartPoleSwingUp-v0, CartPoleSwingUp-v1
    # If you have PyTorch installed:
    # TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
    def __init__(self, offset=2, min_range=-10, max_range=10):
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset
        self.org_env = gym.make("CartPoleSwingUp-v0")

    def reset(self):
        return self.org_env.reset()

    def step(self,action):
        # observation = [x_pos, x_dot, np.cos(theta),np.sin(theta),theta_dot]
        observation, reward, done, info = self.org_env.step(action)
        pole_angle = math.atan(observation[3]/observation[2])

        if pole_angle > self.min_range and pole_angle < self.max_range:
            fake_observation = pole_angle + uniform(-self.offset,self.offset)
            observation[4] = fake_observation
            info["uncertain"] = True
        else:
            observation[4] = pole_angle
            info["uncertain"] = False

        return observation, reward, done, info

if __name__ == "__main__":
    env = SwingUpWrapper()
    done = False
    env.reset()

    while not done:
        action = env.org_env.action_space.sample()
        obs, rew, done, info = env.step(action)
        env.org_env.render()