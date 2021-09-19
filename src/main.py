
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
import agents
from cli import run_cli_cmnds
from src import usuc


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


def main():
    model_name = "blitz50k"
    model = neural_net_blitz.load(f"../models/{model_name}.pt")
    #
    # env = discrete_env_with_nn(rf.right, model)
    # from gym_cartpole_swingup.envs import cartpole_swingup
    # env = cartpole_swingup.CartPoleSwingUpV1()
    env = usuc.USUCDiscreteEnv(num_actions=100, noise_offset=0, noisy_circular_sector=(0, 1))
    ppo = agents.create("ppo", env)
    agents.train(ppo, total_timesteps=80000)
    agents.save(ppo, "../agents/ppo")

    input("continue?")
    histories = agents.run(ppo, env, 10)

    with open("history_ppo.p", "wb") as f:
        dill.dump(histories, f)

    evaluation.plot_angles(histories[0], model_name)


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
    #main()
    #analysis()
    run_cli_cmnds()
