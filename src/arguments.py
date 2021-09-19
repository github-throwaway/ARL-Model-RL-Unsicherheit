import argparse

def collect_arguments():
    def str2bool(s):
        if s == False or s.lower() in ["false", "f", "0"]:
            return False
        return True

    def str2list(s):
        return list(map(int, s.split()))

    parser = argparse.ArgumentParser(
        description='Define the parameters for USUC-Project')

    # startup arguments
    parser.add_argument("--generate_dataset",     default=False,  type=str2bool,       help="Set to True, if you want to generate a dateset.")
    parser.add_argument("--data_dir",     default="../discrete-usuc-dataset/",  type=str,       help="Define a folder, where you want to save/load your dataset to/from.")
    parser.add_argument("--num_actions",       default=10,            type=int,       help='Define the size of the action space of the environment')
    parser.add_argument("--noise_offset",       default=0.3,            type=int,       help='Define the noise')
    parser.add_argument("--runs",       default=9,            type=int,       help='Define how many runs are used for the dataset')
    parser.add_argument("--time_steps",       default=4,            type=int,       help='Define how many time_steps are used to predict an angle')

    parser.add_argument("--train_nn",     default=False,  type=str2bool,       help="Set to True, if you want to train the neuronal network. Make sure, you also set the data paths")
    parser.add_argument("--nn_folder",     default="./model/",  type=str,       help="Define a folder, where you want to save/load your neuronal network to/from.")

    parser.add_argument("--train_ppo",     default=False,  type=str2bool,       help="Set to True, if you want to train PPO with your neuronal network.")
    parser.add_argument("--ppo_folder",     default="./ppo/",  type=str,       help="Define a folder, where you want to save/load your trained ppo agent to/from.")
    parser.add_argument("--env_name",     default="USUCEnvWithNN",  type=str,       help="Define the environment. USUCEnv-v0 | USUCEnv-v1 | USUCEnvWithNN-v0")

    parser.add_argument("--first_run",     default=False,  type=str2bool,       help="Set to True, if it´s your first run and you want to prepare everything.")

    parser.add_argument("--test",     default=False,  type=str2bool,       help="Set to True, if you want to test your neuronal network.")

    parser.add_argument("--eval",     default=False,  type=str2bool,       help="Set to True, if you want to evaluate your neuronal network.")

    # program arguments
    parser.add_argument("-r", "--render",     default=False,          type=str2bool,  help="Define if the env should render")
    parser.add_argument("-l", "--load",         default=False,           type=str2bool, help="Load the weights for the net from disk")
    parser.add_argument("-m", "--mode",         default="test",        type=str,       help='Mode to evaluate (train|test)')
    parser.add_argument("-e", "--epochs",     default=1000,          type=int,       help="Define the number of epochs for the neuronal net")

    # RL Algorithm Parameters
    parser.add_argument("-a", "--algorithm",    default="ppo",         type=str,       help="Define the algorithm ppo | a2c")
    parser.add_argument("--episodes",     default=50000,          type=int,       help="Define the number of episodes")
    parser.add_argument("--gamma",              default=0.995,          type=float,     help="Gamma")
    parser.add_argument("--batch_size",         default=64,           type=int,       help="timesteps_per_batch")
    parser.add_argument("--learn-rate",         default=3e-4,           type=int,       help="The learning rate for the chosen algorithm")

    return parser.parse_args()