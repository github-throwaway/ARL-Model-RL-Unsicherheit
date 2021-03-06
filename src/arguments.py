import argparse


def str2bool(s):
    if s == False or s.lower() in ["false", "f", "0"]:
        return False
    return True


def str2list(s):
    return list(map(int, s.split()))


def collect_arguments():

    parser = argparse.ArgumentParser(
        description="Define the parameters for USUC-Project"
    )

    # startup arguments
    parser.add_argument(
        "-m",
        "--mode",
        default="demo",
        type=str,
        help="Select a mode and start the corresponding routine:"
             "\n'gen_data': Generate a Dataset"
             "\n\t(Optional Arguments: --num_actions, --noise_offset, --data_dir, --runs, --time_steps)"
             "\n'train_rl': Train a RL agent on a given environment"
             "\n\t(Optional Arguments: --nn_model, --reward, --algorithm, --agent, --train_steps)"
             "\n'eval': Evaluate a trained RL-Agent on a given environment"
             "\n\t(Optional Arguments: --nn_model, --reward, --algorithm, --agent)"
             "\n'demo': Watch a demonstration of the Project",
    )

    parser.add_argument(
        "--env_name",
        default="USUCEnvWithNN-v0",
        type=str,
        help="Define the environment: USUCEnv-v0 | USUCEnv-v1 | USUCEnvWithNN-v0",
    )
    parser.add_argument(
        "--data_dir",
        default="../discrete-usuc-dataset/",
        type=str,
        help="Define a folder, where you want to save/load your dataset to/from.",
    )
    parser.add_argument(
        "--num_actions",
        default=10,
        type=int,
        help="Define the size of the action space of the environment",
    )
    parser.add_argument(
        "--noise_offset", default=0.3, type=int, help="Define the noise"
    )
    parser.add_argument(
        "--runs",
        default=9,
        type=int,
        help="Define how many runs are used for the dataset",
    )
    parser.add_argument(
        "--time_steps",
        default=4,
        type=int,
        help="Define how many time_steps are used to predict an angle",
    )

    parser.add_argument(
        "--nn_model",
        default="blitz5k",
        type=str,
        help="Define the filename of the model you want to save/load.",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        default="ppo",
        type=str,
        help="Define the algorithm ppo | a2c",
    )
    parser.add_argument(
        "--agent",
        default="75k_cos_uncert",
        type=str,
        help="Define the name of the agent you want to save/load.",
    )
    parser.add_argument(
        "--reward",
        default="cos_uncert_light",
        type=str,
        help="Define which reward-function should be used. xpos_theta_uncert | cos | best",
    )

    # program arguments
    parser.add_argument(
        "-r",
        "--render",
        default=False,
        type=str2bool,
        help="Define if the env should render",
    )
    parser.add_argument(
        "--train_steps",
        default=25000,
        type=int,
        help="Define the number of training steps for the neuronal net",
    )

    # RL Algorithm Parameters
    parser.add_argument(
        "--episodes", default=50000, type=int, help="Define the number of episodes"
    )
    parser.add_argument("--gamma", default=0.995, type=float, help="Gamma")
    parser.add_argument(
        "--batch_size", default=64, type=int, help="timesteps_per_batch"
    )
    parser.add_argument(
        "--learn-rate",
        default=3e-4,
        type=int,
        help="The learning rate for the chosen algorithm",
    )
    parser.add_argument(
        "--overwrite_agent",
        default=False,
        type=str2bool,
        help="Define if an agent can be overwritten",
    )

    return parser.parse_args()
