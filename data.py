import os
import json
import itertools
import usuc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uuid import uuid4
from typing import List
from collections import namedtuple

# path constants
TIME_SERIES_FILEPATH = "/time_series.csv"
META_INFO_FILEPATH = "/meta.json"

# types
Record = namedtuple("Record", "x_pos x_dot theta theta_dot action original_theta uncertain")


def gen_records(env: usuc.USUCEnv, filename: str = None) -> List[Record]:
    """
    Generates CSV file for one run of the USUC env with random actions
    :param env: Initialize env instance
    :param filename: Optional filename where to store observations
    :return generated records (one record includes observations and info data)
    """

    history = usuc.random_actions(env)

    # transform collected
    records = [
        Record(
            x_pos=obs.x_pos,
            x_dot=obs.x_dot,
            theta=obs.theta,
            theta_dot=obs.theta_dot,
            action=info["action"],
            original_theta=info["original_theta"],
            uncertain=info["uncertain"]
        ) for (obs, info) in history]

    if filename:
        df_obs = pd.DataFrame(records)
        df_obs.to_csv(filename, index=False, header=Record._fields)

    return records


def gen_timeseries(records: list, time_steps: int, filename: str = None) -> list:
    """
    Generates CSV file for list of time series generated from observations.
    Use ``gen_observations()`` to generate observations.
    :param observations: List of observations from one run
    :param time_steps: Length of one time series
    :param filename: Optional filename where to store observations
    :return generated list of time series
    """
    # TODO: add warning to be careful <- closely coupled to neural_net

    # use rolling window if size "time_steps" to create time series
    df_obs = pd.DataFrame(records, columns=Record._fields)
    windows = [list(win.values) for win in df_obs.rolling(time_steps)]

    # drop all windows (at the beginning) which do not have the full size
    windows = itertools.dropwhile(lambda w: len(w) < time_steps, windows)

    # flatten
    windows = [list(itertools.chain(*win)) for win in windows]

    # save time series list if filename is provided
    if filename:
        df_obs = pd.DataFrame(windows)
        df_obs.to_csv(filename, header=None)

    return windows


def gen(env, time_steps, runs, data_dir):
    """"
    Generates USUC dataset
    """
    time_series_list = []
    for _ in range(runs):
        filename = data_dir + "/" + str(uuid4().time_low)
        env.reset(usuc.random_start_theta())
        records = gen_records(env, filename + "-rec.csv")

        # plot theta progression
        original_angles = [r.original_theta for r in records]
        observed_angles = [r.theta for r in records]
        plot_angles(original_angles, observed_angles, filepath=filename + ".png", show=False)

        # generate time series list from observations of current run
        time_series_list.extend(gen_timeseries(records, time_steps))

    # save time series list
    df_tsl = pd.DataFrame(time_series_list)
    df_tsl.to_csv(data_dir + TIME_SERIES_FILEPATH, index=False, header=None)

    # save meta info
    with open(data_dir + META_INFO_FILEPATH, "w") as json_file:
        meta = {
            "number_of_time_series": len(time_series_list),
            "number_of_runs": runs,
            "time_steps": time_steps,
            "noisy_circular_sector": env.ncs,
            "noise_offset": env.noise_offset,
        }
        json.dump(meta, json_file)


def load(data_dir: str):
    """
    Loads a training and test set from the disk
    :return: ((x_train, y_train), (x_test, y_test))
    """

    # load meta info
    with open(data_dir + META_INFO_FILEPATH) as json_file:
        config = json.load(json_file)
        print("Using config:", config)

    time_steps = config["time_steps"]

    # load time series list
    dataset = np.genfromtxt(data_dir + TIME_SERIES_FILEPATH, delimiter=',').astype(np.float32)

    # split datasets into x and y (where y is the last observation)
    row_len = len(dataset[0])
    obs_len = int(row_len / time_steps)

    # length check for debugging
    assert row_len % time_steps == 0, "Length of the row is not a multiple of the length of one observation"

    split_idx = obs_len * (time_steps - 1)
    x, y = np.hsplit(dataset, [split_idx])

    # currently we only use x_pos and theta in y
    # -> remove columns x_dot (idx: 1) and theta_dot (idx: 3) in y
    y = np.delete(y, [1, 3], 1)

    return x, y


def plot_angles(original: List[float], observed: List[float], filepath: str = None, show=True) -> None:
    """
    Plots the original angles as well as the observed angle in one figure for comparison

    :param original: The original angles
    :param observed: The observed angles including noise (i.e. with uncertainty)
    :param filepath: Optional filepath where figure is saved
    """

    assert len(original) == len(observed), "Length of the lists do not match"

    fig = plt.figure(figsize=(19, 12))
    plt.title("Angle Progression")
    plt.xlabel("Time")
    plt.ylabel("Pole Angle")

    plt.plot(original, 'x', label='Original Angle', color="blue")
    plt.plot(observed, 'x', label='Observed Angle', color="orange")

    plt.legend()
    plt.grid()

    if show:
        plt.show()

    if filepath:
        plt.savefig(filepath)

    plt.close(fig)


if __name__ == '__main__':
    env = usuc.USUCDiscreteEnv(num_actions=10, noise_offset=0.5, render=False)
    time_steps = 5
    runs = 200
    data_dir = "./discrete-usuc"

    # creating dir
    os.mkdir(data_dir)

    gen(env, time_steps, runs, data_dir)
