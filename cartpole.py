import math
import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleEnv(CartPoleEnv):
    """Customized cartpole environment
    TODO: eventually replace with env mentioned by mr. tokic
    """

    def __init__(self):
        super().__init__()

        self.theta_threshold_radians = math.inf

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        pole_angle = self.state[2]
        print(observation)

        return observation, reward, done, _

    def reset(self):
        '''This cart pole should start facing downwards. So we need to add pi to the initial pole position (rad) '''
        super().reset()
        self.state[2] += math.pi

        return np.array(self.state)


if __name__ == '__main__':
    env = CartPoleEnv()

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print(observation)

        if done:
            break

    env.close()
