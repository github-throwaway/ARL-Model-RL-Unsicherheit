import arguments
import reward_functions as rf
from src import usuc, neural_net_blitz
from utils import discrete_env_with_nn
import data
import agents
import dill
import evaluation

args = arguments.collect_arguments()


def run_cli_cmnds():
    if args.reward == "best":
        reward_fn = rf.best
    elif args.reward == "cos":
        reward_fn = rf.cos
    else:
        reward_fn = rf.simple

    if args.env_name == "USUCEnv-v0":
        env = usuc.USUCDiscreteEnv(num_actions=args.num_actions, noise_offset=0, noisy_circular_sector=(0, 1))
    elif args.env_name == "USUCEnv-v1":
        env = usuc.USUCEnv(noise_offset=0, noisy_circular_sector=(0, 1))
    elif args.env_name == "USUCEnvWithNN-v0":
        model = neural_net_blitz.load(f"../models/{args.nn_model}.pt")
        env = discrete_env_with_nn(reward_fn, model)


    if args.mode == "gen_data":
        data.generate_dataset(num_actions=args.num_actions, noise_offset=args.noise_offset, data_dir=args.data_dir,runs=args.runs, time_steps=args.time_steps)
    elif args.mode == "train_env":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "train_rl":
        time_sequences, config = data.load(args.data_dir)
        rl_alg = agents.create(args.algorithm, env)
        agents.train(rl_alg, total_timesteps=80000)
        agents.save(rl_alg, f"../agents/{args.algorithm}_{args.agent}")
        input("continue?")
        histories = agents.run(rl_alg, env, 10)

        with open("history_ppo.p", "wb") as f:
            dill.dump(histories, f)

        evaluation.plot_angles(histories[0], args.nn_model)

    elif args.mode == "test":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "eval":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "first_run":
        data.generate_dataset(num_actions=args.num_actions, noise_offset=args.noise_offset, data_dir=args.data_dir,runs=args.runs, time_steps=args.time_steps)
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "presentation":
        time_sequences, config = data.load(args.data_dir)