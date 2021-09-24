import os
from typing import Callable

import dill

import agents
import arguments
import data
import evaluation
import neural_net

# this import is required to be able to load model
from neural_net import BayesianRegressor
import reward_functions as rf
import usuc
import utils
from demo import demo

args = arguments.collect_arguments()


def run_cli_commands():
    """
    Create startup routines
    """
    reward_fn = reward_function()
    env = load_environment(reward_fn)
    choose_mode(env, reward_fn)


def choose_mode(env: usuc.USUCEnv, reward_fn: Callable):
    """
    Toggle between different modes

    :param env: Instantiated environment
    :param reward_fn: Reward function used to calculate reward
    """
    if args.mode == "gen_data":
        data.generate_dataset(
            num_actions=args.num_actions,
            noise_offset=args.noise_offset,
            data_dir=args.data_dir,
            runs=args.runs,
            time_steps=args.time_steps,
        )
    elif args.mode in ("train_env", "test"):
        data.load(args.data_dir)
    elif args.mode == "train_rl":
        train_rl_agent(env)
    elif args.mode == "eval":
        evaluate_rl_agent(env)
    elif args.mode == "plot":
        plot(reward_fn)
    elif args.mode == "demo":
        demo()


def plot(reward_fn):
    """
    Evaluate a given reward function visually

    :param reward_fn: Reward function to plot
    """
    model = neural_net.load(f"../models/{args.nn_model}.pt")
    env = neural_net.USUCEnvWithNN.create(reward_fn, model, "../discrete-usuc-dataset")
    env.reset(1)
    history = utils.random_actions(env)
    evaluation.plot_angles(
        history, args.nn_model, filename=args.nn_model, show=False
    )
    evaluation.plot_reward_angle(history)


def evaluate_rl_agent(env):
    """
    Evaluate a trained RL-Agent on a given environment

    :param env: Environment to evaluate the RL agent on
    """
    if os.path.isfile(f"../agents/{args.algorithm}_{args.agent}.zip"):
        print("Loading model...")
        rl_alg = agents.load(args.algorithm, f"../agents/{args.algorithm}_{args.agent}")
        histories = agents.run(rl_alg, env, 10)
        with open("history_ppo.p", "wb") as f:
            dill.dump(histories, f)
        evaluation.plot_angles(
            histories[0], args.nn_model, filename=args.agent, show=False
        )
    print("No model found...")
    exit(0)


def train_rl_agent(env):
    """
    Train a RL agent on a given environment

    :param env:  Environment to train the RL agent on
    """
    if os.path.isfile(f"../agents/{args.algorithm}_{args.agent}.zip"):
        print("Loading model...")
        rl_alg = agents.load(args.algorithm, f"../agents/{args.algorithm}_{args.agent}")
    else:
        print("Creating model...")
        rl_alg = agents.create(args.algorithm, env)
    agents.train(rl_alg, total_timesteps=args.train_steps)
    agents.save(rl_alg, f"../agents/{args.algorithm}_{args.agent}")
    input("continue?")
    histories = agents.run(rl_alg, env, 10)
    with open("history_ppo.p", "wb") as f:
        dill.dump(histories, f)
    evaluation.plot_angles(
        histories[0], args.nn_model, filename=args.agent, show=False
    )


def load_environment(reward_fn: Callable) -> usuc.USUCEnv:
    """
    Load the environment dependent on the parameter passed

    :param reward_fn: Reward function used to calculate reward
    :return: Instantiated environment
    """
    if args.env_name == "USUCEnv-v0":
        env = usuc.USUCDiscreteEnv(
            num_actions=args.num_actions, noise_offset=0, noisy_circular_sector=(0, 1)
        )
    elif args.env_name == "USUCEnv-v1":
        env = usuc.USUCEnv(noise_offset=0, noisy_circular_sector=(0, 1))
    elif args.env_name == "USUCEnvWithNN-v0":
        model = neural_net.load(f"../models/{args.nn_model}.pt")
        env = neural_net.USUCEnvWithNN.create(
            reward_fn, model, "../discrete-usuc-dataset"
        )
    return env


def reward_function() -> Callable:
    """
    Choose reward function.
    :return: reward function
    """
    if args.reward == "best":
        reward_fn = rf.best
    elif args.reward == "cos":
        reward_fn = rf.cos
    else:
        reward_fn = rf.simple
    return reward_fn
