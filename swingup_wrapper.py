# coding: utf-8
import math
import pandas as pd
from collections import namedtuple
from random import uniform

import gym
import numpy as np
from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpEnv, CartPoleSwingUpV1, CartPoleSwingUpV0


class SwingUpWrapper(gym.Env):

    # Could be one of:
    # CartPoleSwingUp-v0, CartPoleSwingUp-v1
    # If you have PyTorch installed:
    # TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
    def __init__(self, offset=2, min_range=-10, max_range=10):
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset
        self.org_env = CartPoleSwingUpV0()
        self.action_space = gym.Env.action_space
        self.observation_space = gym.Env.observation_space

    def reset(self):
        state = self.org_env.reset()
        State = namedtuple("State", "x_pos x_dot theta theta_dot")
        random_theta = np.random.uniform(0.0, 2.0 * np.pi)
        self.org_env.state = State(state[0], state[1], random_theta, state[4])
        return self.org_env.state

    # todo track position auch zufällig?

    def step(self, action):
        # observation = [x_pos, x_dot, np.cos(theta),np.sin(theta),theta_dot]
        observation, reward, done, info = self.org_env.step(action)
        pole_angle = math.atan(observation[3] / observation[2])

        my_obs = list(observation)
        if self.min_range < pole_angle < self.max_range:
            fake_observation = pole_angle + uniform(-self.offset, self.offset)
            # todo: verrauschten winkel & Winkelgeschwindigkeit ersetzen
            # differenz zwischen winkel von winkelgeschwindigkeit abziehen
            # nur zurück liefern, nicht im original env verändern!
            fake_angular_velocity = observation[4] - (pole_angle - fake_observation)
            my_obs.append(fake_observation)

            reward = 1 - abs(observation[0]) + np.cos(fake_observation)
            info["uncertain"] = True
        else:
            my_obs.append(pole_angle)
            info["uncertain"] = False

            # todo: Reward = cos(theta_noise, x_pos)
            # Reward == Hypotenuse
            reward = 1 - abs(observation[0]) + np.cos(pole_angle)
            # siehe python notebook aus übung

        return my_obs, reward, done, info




if __name__ == "__main__":
    env = SwingUpWrapper()
    done = False
    state = env.reset()
    observations_list = []
    observations = []

    while not done:
        action = env.org_env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if len(observations) >= 5:
            observations.pop(0)
        observations.append(obs)
        print(obs)
        print(info)
        env.org_env.render()
        observations_list.append(observations.copy())

    #columns=['x_pos', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot', 'Fake-Theta']
    df = pd.DataFrame(observations_list)
    df.to_csv("observations.csv")

    #csv_obs = pd.read_csv('observations.csv')
    #print(csv_obs)
    #print("CSV An Stelle 12: ", csv_obs.values[12])
    # csv_obs.values[12][0] returns the index 12, so start with 1
