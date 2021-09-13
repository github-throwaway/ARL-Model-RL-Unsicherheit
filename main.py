import math
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import neural_net
import reward_functions as rf
import usuc
import arguments


def plot(ground_truths, predictions, upper_border=2 * math.pi, lower_border=0):
    # TODO
    data_dir = "./results"
    pred, std_dev = np.hsplit(predictions, [1])
    # Make sure, pred and std_dev are 1-d Arrays
    # TODO Either use np.concatenate or make sure, array is passed with fitting dimensions
    pred = np.concatenate(pred, axis=None)
    std_dev = np.concatenate(std_dev, axis=None)
    fig = plt.figure(figsize=(19, 12))

    plt.plot(ground_truths, label="Ground Truths", color="blue")
    plt.plot(pred, label="Predictions", color="orange")
    """
    higher_dev = [min(p + s, upper_border) for p, s in zip(pred, std_dev)]
    lower_dev = [max(p - s, lower_border) for p, s in zip(pred, std_dev)]
    x = range(len(std_dev))
    plt.fill_between(x, higher_dev, lower_dev, color='yellow', alpha=0.3)
    plt.plot(higher_dev, color='yellow', alpha=0.5)
    plt.plot(lower_dev, color='yellow', alpha=0.5)
    """
    plt.xlabel("Time")
    plt.ylabel("Pole Angle")
    plt.legend()
    plt.grid()
    plt.savefig(data_dir + "/plot.png")
    plt.show()
    plt.close(fig)


def plot_predictions():
    # TODO
    data_dir = "./results"

    # load ground truths
    _, (_, y_test) = gen.load_datasets(data_dir)
    y_test = list(np.delete(y_test, 0, 1).flat)
    print(y_test)

    # load predictions
    predictions = np.genfromtxt(data_dir + "/predictions.csv", delimiter=",")

    # check if data is continuous
    for i in range(len(y_test) - 1):
        if abs(y_test[i] - y_test[i + 1]) > 0.5:
            print(i, y_test[i], y_test[i + 1])

    # plot
    plot(y_test, predictions)


def ppo_test(env, total_timesteps=50000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)

    input("continue?")

    for _ in range(10):
        obs = env.reset()
        for i in range(1000):
            print(f"{i}. Step:")
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

            if done:
                break


def a2c_test(env, total_timesteps=25000):
    from stable_baselines3 import A2C

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)

    input("continue?")

    for _ in range(10):
        obs = env.reset()
        for i in range(1000):
            print(f"{i}. Step:")
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

            if done:
                break


def ppo_nn_env():
    nn = neural_net.load()
    env = gym.make(
        usuc.USUCEnvWithNN.ID, nn=nn, random_actions=100, reward_fn=rf.simple
    )

    ppo_test(env)
    # a2c_test(env)


def nn_test():
    nn = neural_net.load()
    env = usuc.USUCEnvWithNN(
        nn=nn,
        random_actions=100,
        reward_fn=rf.simple,
        noise_offset=0,
    )

    env.reset()
    history = usuc.random_actions(env)
    print(history[0])


def ppo_original_env():
    from gym_cartpole_swingup.envs.cartpole_swingup import (
        CartPoleSwingUpV1 as CartPoleSwingUp,
    )

    env = CartPoleSwingUp()
    ppo_test(env)
    # a2c_test(env)


def ppo_keep_centered():
    env = usuc.USUCEnv(reward_fn=rf.centered)
    ppo_test(env)


def ppo_keep_within_boundaries():
    env = usuc.USUCEnv(reward_fn=rf.boundaries)
    ppo_test(env)


def ppo_uncertainty_env():
    env = usuc.USUCEnv()

    ppo_test(env)
    # a2c_test(env)


def ppo_discrete_uncertainty_env():
    env = usuc.USUCDiscreteEnv()

    ppo_test(env)
    # a2c_test(env)


if __name__ == "__main__":
    args = arguments.collect_arguments()
    # x, y = data.load("./usuc")
    usuc.register_envs()

    # ppo_original_env()
    # ppo_keep_centered()
    # ppo_keep_within_boundaries()
    # ppo_uncertainty_env()
    # ppo_discrete_uncertainty_env()
    if args.mode == "test":
        nn_test()
    elif args.mode == "train":
        ppo_nn_env()