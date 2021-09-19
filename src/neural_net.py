from collections import namedtuple
from itertools import chain
from typing import Callable
from uuid import uuid4

import evaluation
import torch
import torch.nn as nn
import torch.optim as optim
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data
import usuc

# types
Prediction = namedtuple("Prediction", "theta_sin, std_sin, theta_cos, std_cos")


@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Simple Bayesian Neural Net"""
        super().__init__()
        self.blinear = BayesianLinear(input_dim, output_dim)

    def forward(self, x):
        return self.blinear(x)


class NeuralNet:
    def __init__(self, model, time_steps, samples):
        """
        Neural net wrapper

        :param model:
        :param time_steps: Number of time steps for input for prediction
        """
        assert time_steps > 0, "Param time_steps must be greater than 0"

        self.model = model
        self.time_steps = time_steps
        self.samples = samples

    def predict(self, history: list, action) -> tuple:
        """
        Predicts sine and cosine parts of the next angle and their stds

        :param history: The n recent observations t0, ..., tn)
        :param action: The current action to transition from tn to tn+1
        :return: predicted angle, predicted std
        """

        # transform recent history and current action to valid input for nn
        time_series = transform(history, action)

        # make prediction
        x = transform(history, action)
        x = torch.tensor(x).float()

        # sample
        means, stds = samples(self.model, x, self.samples)

        # format
        return Prediction(
            theta_sin=means[0][0],
            std_sin=means[0][1],
            theta_cos=stds[0][0],
            std_cos=stds[0][1])


class USUCEnvWithNN(usuc.USUCDiscreteEnv):
    ID = "USUCEnvWithNN-v0"

    def __init__(self, nn, reward_fn: Callable[[usuc.Observation, float, dict], float], *args, **kwargs):
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

    @classmethod
    def create(cls, model, reward_fn, filepath):
        """
        Creates a USUCEnvWithNN from the given parameters

        :param model: Model used for angle prediction
        :param reward_fn: Reward function used to calculate reward
        :param filepath: Filepath of the dataset the model was trained on
        (uses config of the dataset for auto configuration)
        :return: Instantiated USUCEnvWithNN
        """
        _, config = data.load(filepath)

        ncs = config["noisy_circular_sector"]
        noise_offset = config["noise_offset"]
        time_steps = config["time_steps"]
        num_actions = config["num_actions"]

        nn = NeuralNet(model, time_steps, 25)

        return cls(
            nn=nn,
            num_actions=num_actions,
            reward_fn=reward_fn,
            noise_offset=noise_offset,
            noisy_circular_sector=(ncs,),
        )

    def step(self, action_idx):
        observation, reward, done, info = super().step(action_idx)

        # get recent history
        recent_history = self._history[-self.nn.time_steps:]

        # make prediction
        pred = self.nn.predict(recent_history, self.actions[action_idx])
        info.update(
            {
                "observed_theta_sin": observation.theta_sin,
                "observed_theta_cos": observation.theta_cos,
                "predicted_theta_sin": pred.theta_sin,
                "predicted_std_sin": pred.std_sin,
                "predicted_theta_cos": pred.theta_cos,
                "predicted_std_cos": pred.std_cos,
            }
        )

        # update observation with predicted theta
        # check to remind us to update this piece of code if we switch to cos/sin representation
        new_observation = observation._replace(theta_sin=pred.theta_sin, theta_cos=pred.theta_cos)

        # calculate reward based on prediction
        new_reward = self.nn_reward_fn(observation, reward, info)

        # add step to history
        self._history.append((observation, new_reward, done, info))

        return new_observation, new_reward, done, info

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


def transform(history, current_action):
    """
    Transforms given history and current action to time series used as neural net input
    (i.e. a flattened list of values)

    :param history: List of observations
    :param current_action: Action to transition to next observation
    :return: Time series
    """

    time_series = [obs for (obs, _, __, info) in history]
    # flatten
    time_series = list(chain.from_iterable(time_series))
    # append current action
    time_series.append(current_action)

    return time_series


def samples(model, x, samples=100):
    """
    Predicts for each input an output

    :param model: Model used for prediction
    :param x: List of inputs
    :param samples: Number of samples used for prediction
    :return: List of outputs (corresponding to inputs)
    """
    preds = [model(x) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)

    return means, stds


def load_discrete_usuc(size: int = None, test_size=0.25):
    """
    Loads discrete usuc dataset
    (when size is not set, complete dataset is used)

    :param size: Number of lines used for the dataset
    :param test_size: Percentage of the dataset to be used for test data
    :return: (x_train, y_train, x_test, y_test)
    """
    data_dir = "../discrete-usuc-dataset"
    time_sequences, config = data.load(data_dir)

    # instantiate env (used to get continuous actions)
    env = usuc.USUCDiscreteEnv(
        num_actions=config["num_actions"],
        noise_offset=config["noise_offset"],
        noisy_circular_sector=config["noisy_circular_sector"]
    )

    # fit data / create inputs & outputs
    x, y = [], []

    # reduce size of dataset if specified
    if size:
        time_sequences = time_sequences[:size]

    for ts in time_sequences:
        inputs, output = ts[:-1], ts[-1]
        obs, _, __, info = output

        last_action = env.actions[info["action"]]
        x.append(transform(inputs, last_action))
        y.append([obs.theta_sin, obs.theta_cos])

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

    # convert data to tensors
    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_test, y_test = torch.tensor(x_test).float(), torch.tensor(y_test).float()

    # return data as np arrays
    return x_train, y_train, x_test, y_test


def dataloader(x, y):
    """
    Creates and returns a dataloader for the given data

    :param x: List of inputs
    :param y: List of ouputs
    :return: dataloader
    """

    ds = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

    return dataloader


def generate_model(x_train, y_train):
    """
    Generates a new model from the given data

    :param x_train: List of training inputs
    :param y_train: List of training outputs/labels
    :return: Generated model
    """
    dataloader_train = dataloader(x_train, y_train)

    regressor = BayesianRegressor(x_train.shape[1], 2)

    optimizer = optim.Adam(regressor.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    complexity_cost_weight = 1. / x_train.shape[0]

    losses = []
    for epoch in tqdm(range(100)):
        new_epoch = True
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            loss = regressor.sample_elbo(
                inputs=datapoints,
                labels=labels,
                criterion=criterion,
                sample_nbr=3,
                complexity_cost_weight=complexity_cost_weight
            )

            loss.backward()
            optimizer.step()

            if new_epoch:
                losses.append(loss)
                new_epoch = False

    # plot losses
    evaluation.plot_losses(losses)

    # save model
    torch.save(regressor, f"../models/blitz-{str(uuid4().time_low)}.pt")


def load(filepath):
    """
    Loads model

    :param filepath: Filepath where model is located
    :return: Loaded model
    """
    return torch.load(filepath)


if __name__ == '__main__':
    x_train, y_train, y_test, y_test = load_discrete_usuc(size=5000)
    generate_model(x_train, y_train)
