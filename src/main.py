from stable_baselines3 import PPO, A2C

import arguments
import data
import neural_net
import reward_functions as rf
import utils
import dill
import evaluation
import neural_net_blitz
from tqdm import tqdm
from neural_net_blitz import BayesianRegressor
import dill
import usuc

def discrete_env_with_nn(reward_fn, model) -> neural_net.USUCEnvWithNN:
    """
    Loads model and the config of the dataset.
    Note: Make sure model is trained on the current dataset

    :return: Initialized env
    """
    _, config = data.load("../discrete-usuc-dataset")
    nn = neural_net.NeuralNet(model, 4, 25)



    ncs = config["noisy_circular_sector"]

    return neural_net.USUCEnvWithNN(
        nn=nn,
        num_actions=config["num_actions"],
        reward_fn=reward_fn,
        noisy_circular_sector=(ncs[0], ncs[1]),
        noise_offset=config["noise_offset"],
        render=True
    )


def ppo(env, total_timesteps, filepath=None):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)
    model.save("../agents/ppo")

    if filepath:
        model.save(filepath)

    return model


def a2c(env, total_timesteps, filepath=None):
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)

    if filepath:
        model.save(filepath)

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

    if args.mode == "gen_data":
        data.generate_dataset(num_actions=args.num_actions, noise_offset=args.noise_offset, data_dir=args.data_dir,runs=args.runs, time_steps=args.time_steps)
    elif args.mode == "train":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "test":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "eval":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "first_run":
        data.generate_dataset(num_actions=args.num_actions, noise_offset=args.noise_offset, data_dir=args.data_dir,runs=args.runs, time_steps=args.time_steps)
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "presentation":
        time_sequences, config = data.load(args.data_dir)


def main():
    model_name = "blitz5k"
    model = neural_net_blitz.load(f"../models/{model_name}.pt")

    env = discrete_env_with_nn(rf.right, model)
    env = usuc.USUCDiscreteEnv(num_actions=10, noise_offset=0, noisy_circular_sector=(0, 1), reward_fn=rf.right)
    agent = ppo(env, total_timesteps=10000, filepath="../agents/ppo")
    input("continue?")
    history = run(env, agent)

    with open("history_ppo.p", "wb") as f:
        dill.dump(history, f)

    evaluation.plot_angles(history, model_name)


def test2():
    model_name = "blitz5k"
    model= neural_net_blitz.load(f"../models/{model_name}.pt")

    env = discrete_env_with_nn(rf.best, model)
    env.reset(1)
    history = utils.random_actions(env)
    evaluation.plot_angles(history, model_name)
    evaluation.plot_reward_angle(history)

def analysis():
    # load history for analysis
    with open("history_ppo.p", "rb") as f:
        history = dill.load(f)

    evaluation.plot_reward_angle(history)


if __name__ == "__main__":
    # test2()
    main()
    analysis()
    # TODO Call cli()
