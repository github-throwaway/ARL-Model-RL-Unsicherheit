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
    parser.add_argument("-g", "--graphics",     default=False,          type=str2bool,  help="Define if the env should render")
    parser.add_argument("-l", "--load",         default=False,           type=str2bool, help="Load the weights for the net from disk")
    parser.add_argument("-m", "--mode",         default="train",        type=str,       help='Mode to evaluate (train|test)')
    parser.add_argument("--noise_offset",       default=0.5,            type=int,       help='Define the noise')

    # RL Algorithm Parameters
    parser.add_argument("-a", "--algorithm",    default="ppo",         type=str,       help="Define the algorithm ppo | a2c")
    parser.add_argument("-e", "--episodes",     default=50000,          type=int,       help="Define the number of episodes")
    parser.add_argument("--gamma",              default=0.995,          type=float,     help="Gamma")
    parser.add_argument("--batch_size",         default=20240,           type=int,       help="timesteps_per_batch")
    
    return parser.parse_args()