# coding: utf-8
import math
from collections import namedtuple
from random import uniform

import numpy as np
from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpEnv


class SwingUpWrapper():
    # Could be one of:
    # CartPoleSwingUp-v0, CartPoleSwingUp-v1
    # If you have PyTorch installed:
    # TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
    def __init__(self, offset=2, min_range=-10, max_range=10):
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset
        self.org_env = CartPoleSwingUpEnv()

    def random_start(self):
        self.org_env.reset()
        state = self.org_env.state
        State = namedtuple("State", "x_pos x_dot theta theta_dot")
        random_theta = np.random.uniform(0.0,2.0*np.pi)
        self.org_env.state = State(state.x_pos,state.x_dot,random_theta,state.theta_dot)

    def step(self,action):
        # observation = [x_pos, x_dot, np.cos(theta),np.sin(theta),theta_dot]
        observation, reward, done, info = self.org_env.step(action)
        pole_angle = math.atan(observation[3]/observation[2])

        my_obs = list(observation)
        if pole_angle > self.min_range and pole_angle < self.max_range:
            fake_observation = pole_angle + uniform(-self.offset,self.offset)
            my_obs.append(fake_observation)
            info["uncertain"] = True
        else:
            my_obs.append(pole_angle)
            info["uncertain"] = False

        return my_obs, reward, done, info

if __name__ == "__main__":
    env = SwingUpWrapper()
    done = False
    env.random_start()

    while not done:
        action = env.org_env.action_space.sample()
        obs, rew, done, info = env.step(action)
        print(obs)
        print(info)
        env.org_env.render()