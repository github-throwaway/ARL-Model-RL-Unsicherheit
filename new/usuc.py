# coding: utf-8
import math
import gym
import random
import itertools
import numpy as np
from collections import namedtuple
from gym import spaces
from typing import Tuple, List, Callable
from gym.envs.registration import register

# Could be one of:
# - CartPoleSwingUp-v0,
# - CartPoleSwingUp-v1
# or If you have PyTorch installed:
# - TorchCartPoleSwingUp-v0,
# - TorchCartPoleSwingUp-v1
from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpV0 as CartPoleSwingUp


class USUCEnv(gym.Env):
    ID = 'USUCEnv-v0'

    # TODO: write error message when env not initilaized via reset
    # TODO: implement as discrete action space
    def __init__(self,
                 noisy_circular_sector=(0, 0.5 * math.pi),
                 noise_offset=0.1,
                 reward_fn: Callable = lambda obs, reward, info, action: reward,
                 render=True,
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

    @staticmethod
    def register():
        """
        Register USUCEnv
        """
        env_id = USUCEnv.ID
        register(
            env_id,
            entry_point='usuc:USUCEnv',
        )
        print("registered Uncertain SwingUp Cartpole Env as", env_id)

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
        # TODO: move type State out of reset function
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
            theta = math.acos(theta_cos)
        else:
            theta = math.acos(theta_cos * -1) + math.pi

        # TODO: move jump to bottom
        # if observation[3]>0:
        #     pole_angle = (math.acos(observation[2]) + math.pi) % (math.pi*2)
        # else:
        #     pole_angle = (math.acos(observation[2]*-1)+math.pi * 2) % (math.pi*2)

        # if pole angle is noisy circular sector -> create noisy fake angle
        ncs_start, ncs_end = self.noisy_circular_sector
        if ncs_start < theta < ncs_end:
            # create fake angle
            rng = np.random.default_rng()
            noise = rng.normal(scale=self.noise_offset)
            fake_theta = theta + noise

            # check 0 < fake_theta < 2pi and adapt angle if necessary
            # TODO: not necessary when using sin/cos for angle representation
            # TODO: is this even necessary?? <- couldnt we use case 4 of the regression tutorial to find the rads which are noisy?
            if fake_theta > 2 * math.pi:
                fake_theta -= 2 * math.pi
            elif fake_theta < 0:
                fake_theta += 2 * math.pi

            # adapt theta_dot
            fake_theta_dot = theta_dot + noise * 100 - (theta - fake_theta)

        else:
            fake_theta = None
            fake_theta_dot = None

        # build new observation
        new_observation = [
            x_pos,
            x_dot,
            fake_theta if fake_theta else theta,
            fake_theta_dot if fake_theta_dot else theta_dot
        ]

        # calculate reward
        new_reward = self.reward_fn(new_observation, reward, info, action)

        # update info object
        info["uncertain"] = fake_theta is not None
        info["original_theta"] = theta
        info["action"] = action

        # logging
        if self.verbose:
            print("=== step ===")
            print("action:", action)
            print("original theta:", theta)
            print("fake angle", fake_theta)
            print("observation:", new_observation)

        return new_observation, new_reward, done, info

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
    ID = 'USUCEnvWithNN-v0'

    # TODO: rename var time_steps -> find better name
    # TODO: observation as dict (is supported by ppo in stabel baseline)
    def __init__(self, nn, time_steps, step=0.1, reward_fn=None, **kwargs):
        super().__init__(**kwargs)

        # check step size
        assert self.action_space.low[0] < step < self.action_space.high[0], "Step size exceeds interval size"

        # configuration
        self.nn = nn
        self.reward_fn_with_nn = reward_fn
        self.history = []
        self.time_steps = time_steps

        # convert continuous action space to discrete action space (use gym classes)
        lower_bound = self.action_space.low[0]
        upper_bound = self.action_space.high[0]
        self.actions = list(np.arange(lower_bound, upper_bound, step))
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def register():
        """
        Register USUCEnvWithNN
        """
        env_id = USUCEnvWithNN.ID
        register(
            env_id,
            entry_point='usuc:USUCEnvWithNN',
        )
        print("registered Uncertain SwingUp Cartpole Env with Neural Network as", env_id)

    def step(self, action_index):
        # map action index to action
        action = self.actions[action_index]

        # function to reorder values to feed neural network with (needs special order)
        reorder_values = lambda x_pos, x_dot, theta, theta_dot, action: [x_pos, x_dot, theta_dot, action, theta]

        # execute step by underlying env
        observation, reward, done, info = super().step(action)
        x_pos, x_dot, theta, theta_dot = observation

        # get recent history
        recent_history = self.history[-self.time_steps:]

        # build time series from recent history
        time_series = list(itertools.chain.from_iterable([reorder_values(*obs, action) for (obs, action) in recent_history]))

        # append given action to time series
        time_series.append(action)

        # make prediction
        predicted_theta, predicted_std = self.nn.predict(time_series)

        # build return values
        predicted_observation = [x_pos, x_dot, predicted_theta, theta_dot]
        new_info = {
            **info,
            "predicted_theta": predicted_theta,
            "predicted_std": predicted_std
        }

        # calculate reward based on prediction
        # TODO: install black
        # TODO: define type for reward fn (also in USUCEnv)
        # TODO: define types
        # TODO: reward = reward * (1 - uncertainty)
        new_reward = self.reward_fn_with_nn(predicted_observation, reward, new_info, action)

        return predicted_observation, new_reward, done, new_info

    def reset(self, start_angle: float = None, start_pos: float = None) -> tuple:
        # reset
        state = super().reset(start_angle, start_pos)

        # collect first observations
        for _ in range(self.time_steps):
            action = random.choice(self.actions)
            observation, _, _, info = super().step(action)
            self.history.append((observation, action))

        return state


def register_envs() -> None:
    """
    Registers the gyms USUCEnv and USUCEnvWithNN
    """

    USUCEnv.register()
    USUCEnvWithNN.register()


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
