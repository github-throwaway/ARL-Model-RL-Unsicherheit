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
    
    # program arguments
    parser.add_argument("-n", "--env_name",     default="USUCEnv-v0",  type=str,       help="Define the environment. USUCEnv-v0 | USUCEnv-v1 | USUCEnvWithNN-v0")
    parser.add_argument("-r", "--render",     default=False,          type=str2bool,  help="Define if the env should render")
    parser.add_argument("-l", "--load",         default=False,           type=str2bool, help="Load the weights for the net from disk")
    parser.add_argument("-m", "--mode",         default="test",        type=str,       help='Mode to evaluate (train|test)')
    parser.add_argument("--noise_offset",       default=0.5,            type=int,       help='Define the noise')
    parser.add_argument("-e", "--epochs",     default=1000,          type=int,       help="Define the number of epochs for the neural net")
    parser.add_argument("-t", "--timesteps",     default=4,          type=int,       help="Define the number of timesteps for the neural net")

    # RL Algorithm Parameters
    parser.add_argument("-a", "--algorithm",    default="ppo",         type=str,       help="Define the algorithm ppo | a2c")
    parser.add_argument("--episodes",     default=50000,          type=int,       help="Define the number of episodes")
    parser.add_argument("--gamma",              default=0.995,          type=float,     help="Gamma")
    parser.add_argument("--batch_size",         default=64,           type=int,       help="timesteps_per_batch")
    parser.add_argument("--learn-rate",         default=3e-4,           type=int,       help="The learning rate for the chosen algorithm")

    return parser.parse_args()