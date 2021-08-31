import usuc
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def test():
    usuc.register()

    # Parallel environments

    #env = make_vec_env("USUCEnv-v0", n_envs=1)
    env = gym.make("USUCEnv-v0", render=True, noise_offset=0)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save("ppo_cartpole")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

if __name__ == '__main__':
    test()