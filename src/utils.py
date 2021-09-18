import math
from typing import List

import gym
import numpy as np


def calc_theta(sin, cos):
    """
    Calculates returns theta given by its sine and cosine parts

    :param sin: Sine part of theta
    :param cos: Cosine part of theta
    :return: Theta
    """
    # transform theta_sin, theta_cos to theta (rad)
    if sin > 0:
        theta = math.acos(cos)
    else:
        theta = math.acos(cos * -1) + math.pi

    return theta


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
        obs, reward, done, info = env.step(action)

        # store data
        history.append((obs, reward, done, info))

        if done:
            break

    return history
