import math

import dill

import agents
import evaluation
import neural_net
# this import is required to be able to load model
from neural_net import BayesianRegressor
import reward_functions as rf
import usuc
import utils

from cli import run_cli_cmnds


def test():
    # env = discrete_env_with_nn(rf.right, model)
    # from gym_cartpole_swingup.envs import cartpole_swingup
    # env = cartpole_swingup.CartPoleSwingUpV1()
    env = usuc.USUCDiscreteEnv(num_actions=10, noise_offset=0.3, noisy_circular_sector=(math.pi, 2 * math.pi))
    ppo = agents.create("ppo", env)
    agents.train(ppo, total_timesteps=80000)
    agents.save(ppo, "../agents/ppo")

    input("continue?")
    histories = agents.run(ppo, env, 10)

    evaluation.plot_angles(histories[0], "no model")


def main():
    model_name = "blitz5k"
    model = neural_net.load(f"../models/{model_name}.pt")

    env = neural_net.USUCEnvWithNN.create(model, rf.best, "../discrete-usuc-dataset")

    ppo = agents.create("ppo", env)
    agents.train(ppo, total_timesteps=10000)
    agents.save(ppo, "../agents/ppo")

    input("continue?")
    histories = agents.run(ppo, env, 10)

    evaluation.plot_angles(histories[0], model_name)
    evaluation.plot_reward_angle(histories[0])


if __name__ == "__main__":
    # test()
    # main()
    run_cli_cmnds()
