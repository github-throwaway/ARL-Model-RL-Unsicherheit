import math
import numpy as np
from random import uniform
from gym.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class UpswingCartPoleEnv(CartPoleEnv):
    """Customized cartpole environment
    TODO: eventually replace with env mentioned by mr. tokic
    """

    def __init__(self, offset=2, min_range=-10, max_range=10):
        super().__init__()
        self.theta_threshold_radians = math.inf
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset

    def step(self, action):
        observation, reward, done, info = super().step(action)
        pole_angle = self.state[2]
        # todo: should we also implement uncertainty for a region of the
        #  track?

        if pole_angle > self.min_range and pole_angle < self.max_range:
            np.append(observation,pole_angle + uniform(-1 * self.offset,
                                                  self.offset))
            info["uncertain"] = True
        else:
            np.append(observation,pole_angle)
            info["uncertain"] = False

        return observation, reward, done, info

    def reset(self):
        """This cart pole should start facing downwards. So we need to add pi to the initial pole position (rad) """
        super().reset()
        self.state[2] += math.pi

        return np.array(self.state)


if __name__ == "__main__":
    env = UpswingCartPoleEnv()
    env = make_vec_env(type(env), n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save("save_model")
    del model  # remove to demonstrate saving and loading
    model = PPO.load("save_model")

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        print(f"original: {observation[2]} - fake: {observation[4]} - "
              f"diff: {observation[2]-observation[4]}")
        print()

        if done:
            break

    env.close()
