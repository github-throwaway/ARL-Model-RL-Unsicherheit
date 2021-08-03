# coding: utf-8
import math
import pandas as pd
from collections import namedtuple
from random import uniform, choice
from numpy import genfromtxt
import gym
import numpy as np


import tensorflow as tf
keras = tf.keras
K = keras.backend
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
KL = keras.optimizers
import tensorflow_probability as tfp
tfd = tfp.distributions


from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpEnv, CartPoleSwingUpV1, CartPoleSwingUpV0





class SwingUpWrapper(gym.Env):

    # Could be one of:
    # CartPoleSwingUp-v0, CartPoleSwingUp-v1
    # If you have PyTorch installed:
    # TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
    def __init__(self, offset=0.1, min_range=-10, max_range=10):
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset
        self.org_env = CartPoleSwingUpV0()
        self.action_space = gym.Env.action_space
        print("space", self.action_space)
        print("org space", gym.Env.action_space)
        self.observation_space = gym.Env.observation_space
        print("observation space", self.observation_space)

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
        #pole_angle = observation[2]

        my_obs = list(observation)
        if self.min_range < pole_angle < self.max_range:
            fake_observation = pole_angle + uniform(-self.offset, self.offset)
            fake_observation= checkBoundaries(fake_observation)
            # todo: verrauschten winkel & Winkelgeschwindigkeit ersetzen
            # differenz zwischen winkel von winkelgeschwindigkeit abziehen
            # nur zurück liefern, nicht im original env verändern!
            fake_angular_velocity = observation[4] - (pole_angle - fake_observation)
            my_obs.append(fake_observation)


            #reward = 1 - abs(observation[0]) + np.cos(fake_observation)
            reward = -abs(observation[0]) + (observation[2]-1)
            #my_obs.append(reward)
            info["uncertain"] = True
        else:
            my_obs.append(pole_angle)
            info["uncertain"] = False

            # todo: Reward = cos(theta_noise, x_pos)
            # Reward == Hypotenuse
           # reward = 1 - abs(observation[0]) + np.cos(pole_angle)

            reward = -abs(observation[0]) + (observation[2]-1)
           # my_obs.append(reward)
            # siehe python notebook aus übung

        return my_obs, reward, done, info

def checkBoundaries(fake_observation):
    if (fake_observation>math.pi/2) :
        return math.pi/2
    elif (fake_observation < -math.pi/2) :
        return -math.pi/2

    else:
        return fake_observation




if __name__ == "__main__":
    env = SwingUpWrapper()

    #100 durchläufe werden in csv gespeichert
    for _ in range(1):

        done = False
        state = env.reset()
        observations_list = []
        observations = []
        numberOfValuesPerObservation = 4
        numberOfTimeSteps = 5


        while not done:
            action = env.org_env.action_space.sample()
            #action = choice([0])
            #print("space lower", env.org_env.action_space)
            #print("current action", action)
            obs, rew, done, info = env.step(action)
            print(obs)
            #observations.append(obs)
            observations.append(obs[0])
            observations.append(obs[1])
            observations.append(obs[4])
            observations.append(obs[len(obs)-1])
            if len(observations) > numberOfValuesPerObservation*numberOfTimeSteps:
                observations.pop(0)
                observations.pop(0)
                observations.pop(0)
                observations.pop(0)
                observations_list.append(observations.copy())
            #print(obs)
           # print(info)
           # print("space", action_space)
            #env.org_env.render()

        #columns=['x_pos', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot', 'Fake-Theta'] reward
        df = pd.DataFrame(observations_list)
        #df.to_csv("observations.csv", index=None, header=None, mode="a")
        df.drop(df.columns[17], axis=1, inplace=True)
        df.drop(df.columns[17], axis=1, inplace=True)
        df.to_csv("observations.csv", index=None, header=None)




