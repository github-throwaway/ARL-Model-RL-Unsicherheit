import math

from stable_baselines3 import PPO

import arguments
import data
import neural_net
import reward_functions as rf
import usuc


def discrete_env_with_nn(reward_fn) -> usuc.USUCEnvWithNN:
    """
    Loads model and the config of the dataset.
    Note: Make sure model is trained on the current dataset

    :return: Initialized env
    """
    nn = neural_net.load()
    _, config = data.load()

    ncs = config["noisy_circular_sector"]

    return usuc.USUCEnvWithNN(
        nn=nn,
        random_actions=config["num_actions"],
        reward_fn=reward_fn,
        noisy_circular_sector=(ncs[0], ncs[1]),
        noise_offset=config["noise_offset"],
        render=True
    )


def ppo(env, total_timesteps=50000):
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


def ppo_nn_env():
    nn = neural_net.load()
    env = usuc.USUCEnvWithNN(
        nn=nn,
        random_actions=100,
        reward_fn=rf.simple,
        noisy_circular_sector=(0, math.pi),
        noise_offset=0.5,
        render=True
    )

    ppo_test(env)
    # a2c_test(env)


def nn_test():
    import evaluation

    nn = neural_net.load()
    env = usuc.USUCEnvWithNN(
        nn=nn,
        random_actions=100,
        reward_fn=rf.simple,
        noisy_circular_sector=(0, math.pi),
        noise_offset=0,
        render=True
    )

    env.reset(start_theta=0)
    history = usuc.random_actions(env)

    original_angles = [info["original_theta"] for (_, info) in history]
    observed_angles = [info["observed_theta"] for (_, info) in history]
    predicted_angles = [info["predicted_theta"] for (_, info) in history]
    reward = [info["reward"] for (_, info) in history]
    std = [info["predicted_std"] for (_, info) in history]

    evaluation.plot_test(original_angles, observed_angles, predicted_angles, std, reward)



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