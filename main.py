import gym
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import arguments
import gym_marioai
from gym_marioai import levels

# Parallel environments
args = arguments.collect_arguments()

if args.env_name == 'Mario':
    # Start MarioServer in cmd
    os.popen('java -jar marioai/gym-marioai/gym_marioai/server/marioai-server-0.1-jar-with-dependencies.jar')
    # Setup Mario Env
    reward_settings = gym_marioai.RewardSettings(dead=-10000, timestep=0)
    env = gym.make('Marioai-v1', render=True,
                   reward_settings=reward_settings,
                   level_path=levels.cliff_level,
                   # compact_observation=True,
                   # trace_length=3,
                   rf_width=7, rf_height=5
                   )
    # Start Visualization of Demo
    for e in range(100):
        s = env.reset()
        done = False
        total_reward = 0

        while not done:
            a = env.JUMP_RIGHT if random.randint(0,1) % 2 == 0 else env.SPEED_RIGHT
            s, r, done, info = env.step(a)
            total_reward += r

        print(f'finished episode {e}, total_reward: {total_reward}')

    print('finished demo')
    exit()

elif args.env_name == "MyCartPole":
    env = make_vec_env()
else:
    env = make_vec_env(args.env_name, n_envs=4)
save_model = "target/{}_{}".format(args.algorithm, args.env_name)

if args.algorithm == "ppo":
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(args.episodes))
    model.save(save_model)
    del model  # remove to demonstrate saving and loading
    model = PPO.load(save_model)
obs = env.reset()
while True:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
