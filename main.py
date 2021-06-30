import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import arguments


# Parallel environments
args = arguments.collect_arguments()

env = make_vec_env(args.env_name, n_envs=4)
save_model = "target/{}_{}".format(args.algorithm, args.env_name)
if args.algorithm == "ppo":
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(args.episodes))
    model.save(save_model)
    del model # remove to demonstrate saving and loading
    model = PPO.load(save_model)

obs = env.reset()
while True:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)