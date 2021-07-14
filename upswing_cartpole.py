import math
import numpy as np
from random import uniform
from gym.envs.classic_control.cartpole import CartPoleEnv


class UpswingCartPoleEnv(CartPoleEnv):
    """Customized cartpole environment
    TODO: eventually replace with env mentioned by mr. tokic
    """

    def __init__(self, offset, min_range, max_range):
        super().__init__()
        self.theta_threshold_radians = math.inf
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset

    def step(self, action):
        observation, reward, done, info = super().step(action)
        pole_angle = self.state[2]

        if pole_angle > self.min_range and pole_angle < self.max_range:
            fake_observation = pole_angle + uniform(-1 * self.offset, self.offset)
            info["uncertain"] = True
        else:
            fake_observation = pole_angle
            info["uncertain"] = False

        return observation, fake_observation, reward, done, info

    def reset(self):
        """This cart pole should start facing downwards. So we need to add pi to the initial pole position (rad) """
        super().reset()
        self.state[2] += math.pi

        return np.array(self.state)


if __name__ == "__main__":
    env = UpswingCartPoleEnv(offset=2, min_range=-10, max_range=10)

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = (env.action_space.sample())
        observation, fake_observation, reward, done, info = env.step(action)
        print(f"original: {observation[2]} - fake: {fake_observation} - "
              f"diff: {observation[2]-fake_observation}")
        print()

        if done:
            break

    env.close()
