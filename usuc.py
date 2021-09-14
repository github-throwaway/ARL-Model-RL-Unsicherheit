# coding: utf-8
import math
from collections import namedtuple
from pprint import pformat
from typing import List, Callable

import gym
import numpy as np
from gym.envs.registration import register
from gym.spaces import Box, Discrete
# Could be one of:
# - CartPoleSwingUp-v0,
# - CartPoleSwingUp-v1
# or If you have PyTorch installed:
# - TorchCartPoleSwingUp-v0,
# - TorchCartPoleSwingUp-v1
from gym_cartpole_swingup.envs import cartpole_swingup

# types
# TODO: when using cos and sin as angle representation -> adapt reset and step function
Observation = namedtuple("observation", "x_pos x_dot theta theta_dot")

# TODO: add docs
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
            noisy_circular_sector=(0, 0.5 * math.pi),
            noise_offset=0.1,
            reward_fn: Callable = lambda obs, reward, info, action: reward,
            render=True,
            verbose=False,
    ):
        """
        **Uncertain SwingUp Cartpole Environment**

        # TODO: rework text, document and test -> how exactly can noise_offset be described?

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

        self.ncs = (ncs_start, ncs_end)
        self.noise_offset = noise_offset

        # env configuration
        self._render = render
        self.verbose = verbose
        self.reward_fn = reward_fn
        self.initialized = False

        # wrapped env references
        self.wrapped_env = cartpole_swingup.CartPoleSwingUpV1()
        self.action_space = self.wrapped_env.action_space

        # overwrite observation space since we use only 4 dims: x_pos, x_dot, theta, theta_dot
        dims = len(Observation._fields)
        high = np.array([np.finfo(np.float32).max] * dims, dtype=np.float32)
        self.observation_space = Box(low=-high, high=high)

    def step(self, action):
        assert (
            self.initialized
        ), "Env is not yet initialized, run env.reset() to initialize"

        obs, rew, done, info = self.wrapped_env.step(action)
        x_pos, x_dot, theta_cos, theta_sin, theta_dot = obs

        # transform theta_sin, theta_cos to angle (rad)
        # TODO: document function calls
        if theta_sin > 0:
            theta = math.acos(theta_cos)
        else:
            theta = math.acos(theta_cos * -1) + math.pi

        # TODO: move jump to bottom
        # if observation[3]>0:
        #     pole_angle = (math.acos(observation[2]) + math.pi) % (math.pi*2)
        # else:
        #     pole_angle = (math.acos(observation[2]*-1)+math.pi * 2) % (math.pi*2)

        # if pole angle is in noisy circular sector -> create noisy fake angle
        fake_theta = None
        if self.ncs[0] < theta < self.ncs[1]:
            # create fake angle
            rng = np.random.default_rng()
            noise = rng.normal(scale=self.noise_offset)
            fake_theta = theta + noise

            # check 0 < fake theta < 2pi and adapt fake theta if necessary
            # TODO: not necessary when using sin/cos for angle representation
            if not 0 <= fake_theta <= 2 * math.pi:
                fake_theta = fake_theta - 2 * math.pi * np.sign(fake_theta)

            # adapt theta_dot
            fake_theta_dot = theta_dot + noise * 100 - (theta - fake_theta)

        # build new observation
        observation = Observation(
            x_pos=x_pos,
            x_dot=x_dot,
            theta=fake_theta if fake_theta else theta,
            theta_dot=fake_theta_dot if fake_theta else theta_dot,
        )

        # calculate reward
        reward = self.reward_fn(observation, rew, info, action)

        # update info object
        info.update(
            {
                "uncertain": fake_theta is not None,
                "original_theta": theta,
                "action": action,
            }
        )

        # logging
        if self.verbose:
            print("=== step ===")
            print("action:", action)
            print("original theta:", theta)
            print("fake angle", fake_theta)
            print("observation:", observation)

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
            start_pos if start_pos else x_pos,
            x_dot,
            start_theta if start_theta else theta,
            theta_dot,
        )

        if self.verbose:
            print("initial state:", self.wrapped_env.state)

        return self.wrapped_env.state

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
    # TODO: add doc this is the discrete env

    ID = "USUCEnv-v1"

    # TODO: method signature
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # convert continuous action space to discrete action space
        lower_bound = self.action_space.low[0]
        upper_bound = self.action_space.high[0]
        step = abs(lower_bound - upper_bound) / num_actions
        self.actions = list(np.arange(lower_bound, upper_bound, step))
        self.action_space = Discrete(len(self.actions))

    def step(self, action_index):
        # map action index to action
        action = self.actions[action_index]

        return super().step(action)


class USUCEnvWithNN(USUCDiscreteEnv):
    # TODO: add doc this is the discrete env

    ID = "USUCEnvWithNN-v0"

    # TODO: rework function signatures regarding params (e.g. init)
    def __init__(self, nn, random_actions, reward_fn=None, **kwargs):
        super().__init__(random_actions, **kwargs)

        # configuration
        self.nn = nn
        self.history = []

        # reward function stored separately since its type differs as it is predicted values of nn to calc the reward
        # TODO: define type for reward fn (also in USUCEnv)
        # TODO: define types
        self.nn_reward_fn = reward_fn

    def step(self, action):
        observation, reward, done, info = super().step(action)
        recent_history = self.history[-self.nn.time_steps:]

        # make prediction
        # TODO: use values not the array! (example: 'predicted_std': array(18.236517, dtype=float32))
        predicted_theta, predicted_std = self.nn.predict(recent_history, action)
        info.update(
            {
                "predicted_theta": predicted_theta,
                "predicted_std": predicted_std,
            }
        )

        # update observation with predicted theta
        # check to remind us to update this piece of code if we switch to cos/sin representation
        assert "theta" in Observation._fields, "Theta is not defined in observation"
        observation = observation._replace(theta=predicted_theta)

        # calculate reward based on prediction
        new_reward = self.nn_reward_fn(observation, reward, info, action)

        return observation, new_reward, done, info

    def reset(self, start_theta: float = None, start_pos: float = None):
        super().reset(start_theta, start_pos)

        # collect initial observations
        # neural network needs multiple observations for prediction
        # i.e. we have pre-fill our history with n observations
        # (where n = self.nn.time_steps)
        for _ in range(self.nn.time_steps):
            action = self.action_space.sample()
            observation, _, _, info = super().step(action)
            self.history.append((observation, info))

        # return last observation as initial state
        return observation


def register_envs() -> None:
    """
    Registers the gyms USUCEnv, USUCDiscreteEnv and USUCEnvWithNN
    """

    USUCEnv.register()
    USUCDiscreteEnv.register()
    USUCEnvWithNN.register()


def random_start_theta() -> float:
    """Returns random start angle"""
    start_theta = np.random.uniform(0.0, 2.0 * np.pi)

    return start_theta


def random_actions(env: gym.Env, max_steps=1000) -> List[tuple]:
    """
    Run env with random actions until max_steps or done = True
    :param env: The env used by the "agent"
    :param max_steps: Max steps for one run
    :return: List of tuples with tuple = (obs, info) for one step
    """

    history = []
    for _ in range(max_steps):
        env.render()

        # take random action
        action = env.action_space.sample()
        obs, _, done, info = env.step(action)

        # store data
        history.append((obs, info))

        if done:
            break

    return history
