from collections import namedtuple
from itertools import chain
from typing import Callable

import torch
from sklearn.model_selection import train_test_split

import data
from usuc import USUCDiscreteEnv, Observation

# types
Prediction = namedtuple("Prediction", "theta_sin, std_sin, theta_cos, std_cos")


class NeuralNet:
    def __init__(self, model, time_steps, samples):
        """

        :param model:
        :param time_steps: Number of time steps for input for prediction
        """
        assert time_steps > 0, "Param time_steps must be greater than 0"

        self.model = model
        self.time_steps = time_steps
        self.samples = samples

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

        # make prediction
        x = transform(recent_history, action)
        x = torch.tensor(x).float()
        pred = self.sample(x)

        return pred

    def sample(self, x):
        preds = [self.model(x) for i in range(self.samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0).detach().numpy()
        stds = preds.std(axis=0).detach().numpy()

        theta_sin = means[0]
        theta_cos = means[1]

        std_sin = stds[0]
        std_cos = stds[1]

        return Prediction(theta_sin, std_sin, theta_cos, std_cos)


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
    time_sequences, config = data.load(data_dir)

    env = usuc.USUCDiscreteEnv(
        num_actions=config["num_actions"],
        noise_offset=config["noise_offset"],
        noisy_circular_sector=config["noisy_circular_sector"]
    )

    # fit data / create inputs & outputs
    x = []
    y = []

    for ts in time_sequences:
        inputs = ts[:-1]
        output = ts[-1]
        obs, _, __, info = output

        last_action = env.actions[info["action"]]
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


def discrete_env_with_nn(reward_fn, model) -> USUCEnvWithNN:
    """
    Loads model and the config of the dataset.
    Note: Make sure model is trained on the current dataset

    :return: Initialized env
    """
    _, config = data.load("../discrete-usuc-dataset")
    ncs = config["noisy_circular_sector"]
    time_steps = config["time_steps"]

    nn = NeuralNet(model, time_steps, 25)

    return USUCEnvWithNN(
        nn=nn,
        num_actions=config["num_actions"],
        reward_fn=reward_fn,
        noisy_circular_sector=(ncs[0], ncs[1]),
        noise_offset=config["noise_offset"],
        render=True
    )
