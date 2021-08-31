# coding: utf-8
import math
import gym
import numpy as np
from collections import namedtuple
from gym import spaces
from typing import Tuple, List, Callable

# Could be one of:
# - CartPoleSwingUp-v0,
# - CartPoleSwingUp-v1
# or If you have PyTorch installed:
# - TorchCartPoleSwingUp-v0,
# - TorchCartPoleSwingUp-v1
from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpV0 as CartPoleSwingUp


class USUCEnv(gym.Env):
    def __init__(self,
                 noisy_circular_sector=(0, 0.5 * math.pi),
                 noise_offset=0.1,
                 reward_fn: Callable = lambda obs, reward: reward,
                 render=False,
                 verbose=False):
        """
        **Uncertain SwingUp Cartpole Environment**

        # TODO: document and test -> how exactly can noise_offset be described?
        # TODO: rework text

        The boundaries of the noisy circular sector (ncs) must be given as radians with ``ncs[0] < ncs[1]``
        (i.e. start and end of the circular sector)

        :param noisy_circular_sector: The circular sector in which the angle should be noisy
        :param noise_offset: TODO: what exactly does it specify? whats its type/possible values and what do they mean? ^^
        :param render: Specifies whether cartpole should be rendered
        :param verbose: Specifies whether additional info should be printed
        """
        # noise configuration
        ncs_start, ncs_end = noisy_circular_sector
        assert 0 <= ncs_start < ncs_end <= 2 * math.pi
        assert noise_offset >= 0
        self.noisy_circular_sector = (ncs_start, ncs_end)
        self.noise_offset = noise_offset

        # performance & logging configuration
        self.should_render = render
        self.verbose = verbose

        # wrapped env references
        self.wrapped_env = CartPoleSwingUp()
        self.action_space = self.wrapped_env.action_space

        # overwrite observation space (we use only 4 dims)
        # TODO: when using cos and sin as angle representation -> adapt reset and step function
        high = np.array([np.finfo(np.float32).max] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high)

        # set reward function
        # TODO: use reward fn in step()
        self.reward_fn = reward_fn

    def seed(self, seed=None):
        self.wrapped_env.seed()

    def reset(self, start_angle: float = None, start_pos: float = None) -> tuple:
        """
        Start pole angle must be given as radians. Use ``math.radians()`` to convert from deg to rad

        :param start_angle: Start pole angle of the cartpole
        :param start_pos: Start position of the cartpole
        :return: Initial observation
        """

        # init wrapped env
        self.wrapped_env.reset()

        # get state
        x_pos, x_dot, theta, theta_dot = self.wrapped_env.state

        # set new values if given
        if start_pos is not None:
            x_pos = start_pos
        if start_angle is not None:
            theta = start_angle

        # update state of wrapped env
        State = namedtuple("State", "x_pos x_dot theta theta_dot")
        self.wrapped_env.state = State(x_pos, x_dot, theta, theta_dot)

        if self.verbose:
            print("initial state:", self.wrapped_env.state)

        return self.wrapped_env.state

    def step(self, action):
        observation, reward, done, info = self.wrapped_env.step(action)
        x_pos, x_dot, theta_cos, theta_sin, theta_dot = observation

        # transform theta_sin, theta_cos to angle (rad)
        if theta_sin > 0:
            pole_angle = math.acos(theta_cos)
        else:
            pole_angle = math.acos(theta_cos * -1) + math.pi

        # if pole angle is noisy circular sector -> create noisy fake angle
        ncs_start, ncs_end = self.noisy_circular_sector
        if ncs_start < pole_angle < ncs_end:
            # create fake angle
            rng = np.random.default_rng()
            noise = rng.normal(scale=self.noise_offset)
            fake_angle = pole_angle + noise

            # check 0 < fake_angle < 2pi and adapt angle if necessary
            # TODO: not necessary when using sin/cos for angle representation
            # TODO: is this even necessary?? <- couldnt we use case 4 of the regression tutorial to find the rads which are noisy?
            if fake_angle > 2 * math.pi:
                fake_angle -= 2 * math.pi
            elif fake_angle < 0:
                fake_angle += 2 * math.pi

            # adapt theta_dot
            # TODO: differenz zwischen dem originalen winkel und dem fake winkel von winkelgeschwindigkeit abziehen
            # fake_angular_velocity = observation[4] - (pole_angle - fake_angle)

        else:
            fake_angle = None

        # build new observation
        new_observation = [
            x_pos,
            x_dot,
            fake_angle if fake_angle else pole_angle,
            theta_dot
        ]

        # calculate reward
        reward = self.reward_fn(new_observation, reward)

        # update info object
        info["uncertain"] = fake_angle is not None
        info["original_angle"] = pole_angle

        # logging
        if self.verbose:
            print("=== step ===")
            print("action:", action)
            print("original angle:", pole_angle)
            print("fake angle", fake_angle)
            print("observation:", new_observation)

        return new_observation, reward, done, info

    def render(self, mode="human", **kwargs) -> None:
        if self.should_render:
            self.wrapped_env.render(mode)

    def close(self):
        self.wrapped_env.close()

    def __str__(self):
        from pprint import pformat

        # print wrapper with default values
        return pformat(vars(self), sort_dicts=False)


class USUCEnvWithNN(USUCEnv):
    def __init__(self, nn):
        super().__init__()

        # set nn
        self.nn = nn
        self.observation_history = []

    def step(self, action):
        new_observation, reward, done, info = super().step(action)
        angle, reward = nn(new_observation)

    def reset(self, start_angle: float = None, start_pos: float = None) -> tuple:
        pass




def register() -> None:
    """
    Registers the gyms USUCEnv and USUCEnvWithNN
    """
    print("registering gym envs...")
    from gym.envs.registration import register

    # USUCEnv registration
    id = 'USUCEnv-v0'
    register(
        id,
        entry_point='usuc:USUCEnv',
    )
    print("registered Uncertain SwingUp Cartpole Env as", id)

    # USUCEnvWithNN registration
    id = 'USUCEnvWithNN-v0'
    register(
        id,
        entry_point='usuc:USUCEnvWithNN',
    )
    print("registered Uncertain SwingUp Cartpole Env with Neural Network as", id)


def random_start_angle() -> float:
    """Returns random start angle"""
    start_angle = np.random.uniform(0.0, 2.0 * np.pi)

    return start_angle


def random_actions(env: gym.Env, max_steps=1000) -> Tuple[List[list], list]:
    """
    Run env with random actions
    :param env: The env used by the "agent"
    :param max_steps: Max steps for one run
    :return: observations, original_angles
    """

    # run 1000 steps or until done=True
    information = []
    observations = []
    for _ in range(max_steps):
        env.render()

        # take random action
        obs, _, done, info = env.step(env.action_space.sample())

        # store data
        information.append(info)
        observations.append(obs)

        if done:
            break

    return observations, information


if __name__ == '__main__':
    print("USUC - Uncertain SwingUp Cartpole")
    print("Default Configuration:", USUCEnv(), sep="\n")
