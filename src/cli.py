import arguments
import reward_functions as rf
import usuc
import data
import agents
import dill
import evaluation
from neural_net import discrete_env_with_nn,BayesianRegressor, load
from demo import demo
import os

args = arguments.collect_arguments()


def run_cli_cmnds():
    """
    Create startup routines
    :return:
    """
    # choose reward function
    if args.reward == "best":
        reward_fn = rf.best
    elif args.reward == "cos":
        reward_fn = rf.cos
    else:
        reward_fn = rf.simple

    # load Env
    if args.env_name == "USUCEnv-v0":
        env = usuc.USUCDiscreteEnv(num_actions=args.num_actions, noise_offset=0, noisy_circular_sector=(0, 1))
    elif args.env_name == "USUCEnv-v1":
        env = usuc.USUCEnv(noise_offset=0, noisy_circular_sector=(0, 1))
    elif args.env_name == "USUCEnvWithNN-v0":
        model = load(f"../models/{args.nn_model}.pt")
        env = discrete_env_with_nn(reward_fn, model)

    if args.mode == "gen_data":
        data.generate_dataset(num_actions=args.num_actions, noise_offset=args.noise_offset, data_dir=args.data_dir,runs=args.runs, time_steps=args.time_steps)
    elif args.mode == "train_env":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "train_rl":
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

        evaluation.plot_angles(histories[0], args.nn_model, filepath=f"plots/{args.agent}", show=False)

    elif args.mode == "test":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "eval":
        time_sequences, config = data.load(args.data_dir)
    elif args.mode == "demo":
        demo()
