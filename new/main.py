import usuc
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import data
import usuc
import neural_net

import reward_functions as rf

def ppo_with_nn():
    nn = neural_net.load()

    env = gym.make(usuc.USUCEnvWithNN.ID, nn=nn, render=True, time_steps=4, reward_fn=rf.simple)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    # model.save("ppo_cartpole")
    # del model  # remove to demonstrate saving and loading
    # model = PPO.load("ppo_cartpole")

    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

def ppo_with_original_env():
    from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpV1 as CartPoleSwingUp

    env = CartPoleSwingUp()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=250000)
    # model.save("ppo_cartpole")
    # del model  # remove to demonstrate saving and loading
    # model = PPO.load("ppo_cartpole")

    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

def run_usuc_with_nn():
    nn = neural_net.load()
    reward_fn = lambda _, reward, __, ___: reward
    env = usuc.USUCEnvWithNN(nn, time_steps=4, reward_fn=reward_fn, render=True)
    env.reset()
    usuc.random_actions(env)


def ppo_cartpole():
    import gym
    env = gym.make('CartPole-v0')

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    # model.save("ppo_cartpole")
    # del model  # remove to demonstrate saving and loading
    # model = PPO.load("ppo_cartpole")

    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()


if __name__ == '__main__':
    # x, y = data.load("./usuc")
    usuc.register_envs()
    ppo_with_original_env()
