import math
from collections import namedtuple
from pprint import pformat
from typing import Callable

import gym
import numpy as np
from gym.envs.registration import register
from gym.spaces import Discrete
from gym_cartpole_swingup.envs import cartpole_swingup

import utils

# types
Observation = namedtuple("observation", "x_pos x_dot theta_sin theta_cos theta_dot")


class USUCEnv(gym.Env):
    ID = "USUCEnv-v0"

    @classmethod
    def register(cls):
        """
        Register Env
        """
        register(
            cls.ID,
            entry_point="usuc:" + cls.__name__,
        )
        print("registered", cls.ID)

    def __init__(
            self,
            noisy_circular_sector,
            noise_offset,
            reward_fn: Callable[[Observation, float, dict], float] = lambda obs, reward, info: reward,
            render=True,
    ):
        """
        **Continuous Uncertain SwingUp Cartpole Environment**

        The boundaries of the noisy circular sector (ncs) must be given as radians with ``ncs[0] < ncs[1]``
        (i.e. start and end of the circular sector)

        :param noisy_circular_sector: The circular sector in which the angle should be noisy
        :param noise_offset: The degree of noise in the noisy area (using normal distribution)
        :param reward_fn: Optional reward function to calculate reward.
        If not specified reward of the wrapped env will be returned
        (params: observation, reward from wrapped env, info object; returns: new reward)

        :param render: Specifies whether cartpole should be rendered
        """

        # noise configuration
        ncs_start, ncs_end = noisy_circular_sector
        assert 0 <= ncs_start <= ncs_end <= 2 * math.pi
        assert noise_offset >= 0

        self.ncs = (ncs_start, ncs_end)
        self.noise_offset = noise_offset

        # env configuration
        self._render = render
        self.reward_fn = reward_fn
        self.initialized = False

        # wrapped env references
        self.wrapped_env = cartpole_swingup.CartPoleSwingUpV1()
        self.action_space = self.wrapped_env.action_space
        self.observation_space = self.wrapped_env.observation_space

    def step(self, action):
        assert (
            self.initialized
        ), "Env is not yet initialized. Run env.reset() to initialize"

        obs, rew, done, info = self.wrapped_env.step(action)
        x_pos, x_dot, original_theta_cos, original_theta_sin, theta_dot = obs

        # transform theta_sin, theta_cos to theta (rad)
        original_theta = utils.calc_theta(original_theta_sin, original_theta_cos)

        # if pole angle is in noisy circular sector -> create noisy fake angle
        fake_theta = None
        if self.ncs[0] < original_theta < self.ncs[1]:
            # create fake angle
            rng = np.random.default_rng()
            noise = rng.normal(scale=self.noise_offset)
            fake_theta = original_theta + noise

            # check 0 < fake theta < 2pi and adapt fake theta if necessary
            if not 0 <= fake_theta <= 2 * math.pi:
                fake_theta = fake_theta - 2 * math.pi * np.sign(fake_theta)

        # transform theta back to theta_sin and theta_cos
        theta = fake_theta if fake_theta else original_theta
        theta_sin = math.sin(theta)
        theta_cos = math.cos(theta)

        # build new observation
        observation = Observation(
            x_pos=x_pos,
            x_dot=x_dot,
            theta_sin=theta_sin,
            theta_cos=theta_cos,
            theta_dot=theta_dot,
        )

        # calculate reward
        reward = self.reward_fn(observation, rew, info)

        # update info object
        info.update(
            {
                "uncertain": fake_theta is not None,
                "observed_theta": theta,
                "original_theta": original_theta,
                "original_theta_sin": original_theta_sin,
                "original_theta_cos": original_theta_cos,
                "action": action,
            }
        )

        return observation, reward, done, info

    def reset(self, start_theta: float = None, start_pos: float = None) -> tuple:
        """
        Start pole angle must be given as radians. Use ``math.radians()`` to convert from deg to rad

        :param start_theta: Start pole angle of the cartpole
        :param start_pos: Start position of the cartpole
        :return: Initial observation
        """
        self.initialized = True
        self.wrapped_env.reset()

        # update state of wrapped env
        State = namedtuple("State", "x_pos x_dot theta theta_dot")
        x_pos, x_dot, theta, theta_dot = self.wrapped_env.state

        self.wrapped_env.state = State(
            start_pos if start_pos is not None else x_pos,
            x_dot,
            start_theta if start_theta is not None else theta,
            theta_dot,
        )

        return Observation(
            x_pos=x_pos,
            x_dot=x_dot,
            theta_sin=math.sin(theta),
            theta_cos=math.cos(theta),
            theta_dot=theta_dot
        )

    def render(self, mode="human", **kwargs) -> None:
        if self._render:
            self.wrapped_env.render(mode)

    def seed(self, seed=None):
        self.wrapped_env.seed()

    def close(self):
        self.wrapped_env.close()

    def __str__(self):
        # print wrapper with default values
        return pformat(vars(self), sort_dicts=False)


class USUCDiscreteEnv(USUCEnv):
    ID = "USUCEnv-v1"

    def __init__(self, num_actions, *args, **kwargs):
        """
        **Discrete Uncertain SwingUp Cartpole Environment**

        Divides the continuous action space of the USUCEnv evenly into `n` discrete actions.
        See :class:`USUCEnv` for additional parameters.

        :param num_actions: The number of actions used as action space
        """
        super().__init__(*args, **kwargs)

        # convert continuous action space to discrete action space
        lower_bound = self.action_space.low[0]
        upper_bound = self.action_space.high[0]
        step = abs(lower_bound - upper_bound) / num_actions
        self.actions = list(np.arange(lower_bound, upper_bound, step))
        self.action_space = Discrete(len(self.actions))

    def step(self, action: int):
        # map action to its continuous action value
        action_value = self.actions[action]

        # calc next step
        observation, reward, done, info = super().step(action_value)

        # overwrite info.action with action_index since action space is discrete
        info.update({"action": action})

        return observation, reward, done, info


def register_envs() -> None:
    """
    Registers the gyms USUCEnv, USUCDiscreteEnv and USUCEnvWithNN
    """

    USUCEnv.register()
    USUCDiscreteEnv.register()
