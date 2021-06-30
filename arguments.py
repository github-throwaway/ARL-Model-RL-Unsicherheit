import argparse

def collect_arguments():
    def str2bool(s):
        if s == False or s.lower() in ["false", "f", "0"]:
            return False
        return True

    def str2list(s):
        return list(map(int, s.split()))

    parser = argparse.ArgumentParser(
        description='Define the parameters for the Project')

    # program arguments
    parser.add_argument("-e", "--episodes",     default=5e4,          type=int,      help="Define the number of episodes")
    parser.add_argument("-g", "--graphics",     default=False,          type=str2bool, help="Define if graphics should be shown")
    parser.add_argument("-a", "--algorithm",    default="ppo",         type=str,      help="Define the algorithm ppo | appo |a2c")
    parser.add_argument("-n", "--env_name",     default="LunarLander-v2",  type=str,      help="Define the environment")
    parser.add_argument("-p", "--port",     default=8123,  type=int,      help="Define the port for mario server")

    return parser.parse_args()