import math

from stable_baselines3 import PPO, A2C

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
    _, config = data.load("../discrete-usuc-dataset")

    ncs = config["noisy_circular_sector"]

    return usuc.USUCEnvWithNN(
        nn=nn,
        num_actions=config["num_actions"],
        reward_fn=reward_fn,
        noisy_circular_sector=(ncs[0], ncs[1]),
        noise_offset=config["noise_offset"],
        render=True
    )


def ppo(env, total_timesteps):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)
    model.save("../agents/ppo")

    return model


def a2c(env, total_timesteps):
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)
    model.save("../agents/a2c")

    return model


def run(env, agent):
    history = []
    for _ in range(10):
        obs = env.reset()
        for i in range(1000):
            print(f"{i}. Step:")
            action, _states = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            history.append((obs, reward, done, info))
            env.render()

            if done:
                break

    return history


def cli():
    args = arguments.collect_arguments()

    # TODO
    # if args.mode == "test":
    #     nn_test()
    # elif args.mode == "train":
    #     ppo_nn_env()


def main():
    import dill
    env = discrete_env_with_nn(rf.simple)
    agent = ppo(env, total_timesteps=500000)
    history = run(agent, env)

    # save history for analysis
    with open("./history.p", "wb") as f:
        dill.dump(history, f)


if __name__ == "__main__":
    main()
