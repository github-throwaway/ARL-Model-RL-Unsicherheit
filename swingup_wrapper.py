# coding: utf-8
import math
import pandas as pd
from collections import namedtuple
from random import uniform
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


            #reward = 1 - abs(observation[0]) + np.cos(fake_observation)
            reward = -abs(observation[0]) + (observation[2]-1)
            my_obs.append(reward)
            info["uncertain"] = True
        else:
            my_obs.append(pole_angle)
            info["uncertain"] = False

            # todo: Reward = cos(theta_noise, x_pos)
            # Reward == Hypotenuse
           # reward = 1 - abs(observation[0]) + np.cos(pole_angle)

            reward = -abs(observation[0]) + (observation[2]-1)
            my_obs.append(reward)
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

    csv_obs = pd.read_csv('observations.csv')
    #print(csv_obs)
    #print("CSV An Stelle 12: ", csv_obs.values[12])
    # csv_obs.values[12][0] returns the index 12, so start with 1
    my_data = genfromtxt('observations1.csv', delimiter=',').astype(np.float32)
    #data1=my_data[35, :].copy()
    #print(data1)
    data1, data2= np.hsplit(my_data, [35])
    #print(data2)
    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)
    model.fit(data1, data2, epochs=500, verbose=False)


    x_tst=tf.expand_dims(data1[1,:],0)
    # Make predictions.
    yhat = model(x_tst)
    print(yhat.mean())

