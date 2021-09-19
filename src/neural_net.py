import math
from collections import namedtuple
from itertools import chain
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import data
from usuc import USUCDiscreteEnv, Observation

# types
PredictedTheta = namedtuple("PredictedTheta", "sin cos")

ACTIONS = USUCDiscreteEnv(num_actions=10, noise_offset=0.3, noisy_circular_sector=(0, math.pi)).actions

class NeuralNet:
    def __init__(self, model, time_steps):
        """

        :param model:
        :param time_steps: Number of time steps for input for prediction
        """
        assert time_steps > 0, "Param time_steps must be greater than 0"

        self.model = model
        self.time_steps = time_steps

    def predict(self, recent_history: list, action) -> tuple:
        """
        TOOD: given action should be the continous action
        Predicts next angle and std for a given time series
        # TODO: reference time_steps set in in init of this NN (used by usuc with NN)
        :param recent_history: The n recent observations t0, ..., tn)
        :param action: The current action to transition from tn to tn+1
        :return: predicted angle, predicted std
        """

        # transform recent history and current action to valid input for nn
        time_series = transform(recent_history, action)

        # magic
        # TODO: was passiert hier?
        x = ""

        # make predictions
        # don't use model.predict here, does not return std
        yhats = self.model(x)
        med = yhats.loc
        std = yhats.scale

        # TODO: log predictions
        # TODO: predict
        # extract values from tensors
        # med = float(backend.eval(np.squeeze(med)))
        # std = float(backend.eval(np.squeeze(std)))

        # med = predicted angle, std = predicted std
        predicted_theta = PredictedTheta(sin=1., cos=med)
        return predicted_theta, std


# TODO: refactor transform()
def transform(recent_history, current_action):
    """
    Transforms given history and current action to time series used as neural net input
    :param observation:
    :param current_action:
    :return: Time series array
    """

    time_series = [obs for (obs, _, __, info) in recent_history]
    # time_series = list(recent_history[-1][0])


    # flatten
    time_series = list(chain.from_iterable(time_series))

    # append current action
    time_series.append(current_action)

    return time_series


def load_discrete_usuc():
    """
    Loads discrete usuc dataset
    :return:
    """
    data_dir = "../discrete-usuc-dataset"
    time_sequences, _ = data.load(data_dir)

    # fit data / create inputs & outputs
    x = []
    y = []

    for ts in time_sequences:
        inputs = ts[:-1]
        output = ts[-1]
        obs, _, __, info = output

        # TODO: action must be normalized via env.actions[info["aciton"]
        last_action = info["action"]
        x.append(transform(inputs, last_action))
        y.append([obs.theta_sin, obs.theta_cos])

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42, shuffle=False)

    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_test, y_test = torch.tensor(x_test).float(), torch.tensor(y_test).float()

    # return data as np arrays
    return x_train, y_train, x_test, y_test


def dataloaders(x_train, y_train, x_test, y_test):
    """
    Preprocesses data to be processable by neural net
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """

    ds_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(x_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=False)

    return dataloader_train, dataloader_test


class USUCEnvWithNN(USUCDiscreteEnv):
    ID = "USUCEnvWithNN-v0"

    def __init__(self, nn, reward_fn: Callable[[Observation, float, dict], float], *args, **kwargs):
        """
        **Discrete Uncertain SwingUp Cartpole Environment with Neural Net**

        Uses neural net to estimate/predict uncertainty of theta which in turn is used to calculate the reward.
        See :class:`DiscreteUSUCEnv` and :class:`USUCEnv` for additional parameters.

        :param nn: The neural net used for estimation/prediction
        :param reward_fn: Required reward function to calculate reward based on predictions of the neural net
        (params: observation, reward from wrapped env, info object; returns: new reward)
        """
        super().__init__(*args, **kwargs)

        # configuration
        self.nn = nn
        self._history = []

        # reward function stored separately since its type differs as it is predicted values of nn to calc the reward
        self.nn_reward_fn = reward_fn

    def step(self, action):
        observation, reward, done, info = super().step(action)

        # get recent history
        recent_history = self._history[-self.nn.time_steps:]

        # make prediction
        predicted_theta, predicted_std = self.nn.predict(recent_history, action)
        info.update(
            {
                "observed_theta_sin": observation.theta_sin,
                "observed_theta_cos": observation.theta_cos,
                "predicted_theta": predicted_theta,
                "predicted_std": predicted_std,
            }
        )

        # update observation with predicted theta
        # check to remind us to update this piece of code if we switch to cos/sin representation
        observation = observation._replace(theta_sin=predicted_theta.sin, theta_cos=predicted_theta.cos)

        # calculate reward based on prediction
        new_reward = self.nn_reward_fn(observation, reward, info)

        # add step to history
        self._history.append((observation, new_reward, done, info))

        return observation, new_reward, done, info

    def reset(self, start_theta: float = None, start_pos: float = None):
        super().reset(start_theta, start_pos)
        self._history = []

        # collect initial observations
        # neural network needs multiple observations for prediction
        # i.e. we have to pre-fill our history with n observations
        # (where n = self.nn.time_steps)
        for _ in range(self.nn.time_steps):
            action = self.action_space.sample()
            observation, reward, done, info = super().step(action)
            self._history.append((observation, reward, done, info))

        # return last observation as initial state
        return observation
