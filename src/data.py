import json
import shutil
from typing import List
from uuid import uuid4

import dill
import gym
from tqdm import tqdm
import os
import math

import usuc
import utils

# path constants
TIME_SEQUENCES_FILEPATH = "/time_sequences.p"
CONFIG_FILEPATH = "/config.json"


def _gen_history(env, filename: str = None) -> List[tuple]:
    """
    Generates history for one run of a USUC env with random actions.
    Each time step is represented as a tuple (observation, info dict).

    :param env: Initialized env instance
    :param filename: Optional filename where to store observations
    :return: History of random run
    """

    history = utils.random_actions(env)

    if filename:
        with open(filename, "wb") as f:
            dill.dump(history, f)

    return history


def _gen_time_sequences(
    history: list, time_steps: int, filename: str = None
) -> List[List[tuple]]:
    """
    Generates list of windows (i.e. time sequence) from history.

    Each window contains t_k, t_(k+1),..., t_(time_steps) elements from history
    where t_(time_steps) is the label/output of the record and t_k, t_(k+1),... are the inputs.

    :param history: History of one run
    :param time_steps: Number of input observations
    :param filename: Optional filename where to store time sequences

    :return: List of time sequences (i.e. windows)
    """

    # use rolling window of size "time_series_length" to create time sequence
    windows = []

    for i in range(time_steps, len(history)):
        windows.append(history[i - time_steps : i + 1])

    # save time sequences if filename is provided
    if filename:
        with open(filename, "wb") as f:
            dill.dump(windows, f)

    return windows


def gen(env: usuc.USUCEnv, runs, time_steps, data_dir) -> None:
    """ "
    Generates USUC dataset

    :param env: USUCEnv instance
    :param runs: Number of env runs from which to collect data (impacts size of the dataset)
    :param time_steps: Number of time steps of the generated time sequences
    :param data_dir: Folder in which the data should be saved

    """

    # generating dataset
    print(
        f"collecting data from {runs} runs (saving history for each run in {data_dir})"
    )
    windows = []
    for _ in tqdm(range(runs)):
        filename = data_dir + "/" + str(uuid4().time_low)
        # TODO: random angle should start between [0, pi/2] or [3/2pi, 2pi] to collect more data in upper half of the circle
        env.reset(utils.random_start_theta())
        history = _gen_history(env, filename + "-rec.p")

        # generate time sequences from history of current run
        windows.extend(_gen_time_sequences(history, time_steps))

    # save time sequences
    time_sequences_file = data_dir + TIME_SEQUENCES_FILEPATH
    print(f"saving time sequences to {time_sequences_file} (this may take a while)")
    with open(time_sequences_file, "wb") as f:
        dill.dump(windows, f)

    # save meta info
    meta_info_file = data_dir + CONFIG_FILEPATH
    print(f"saving meta info to {meta_info_file}")
    with open(meta_info_file, "w") as json_file:
        meta = {
            "num_time_sequences": len(windows),
            "num_actions": env.action_space.n
            if isinstance(env.action_space, gym.spaces.Discrete)
            else None,
            "number_of_runs": runs,
            "time_steps": time_steps,
            "noisy_circular_sector": env.ncs,
            "noise_offset": env.noise_offset,
        }
        json.dump(meta, json_file)


def load(data_dir: str) -> tuple:
    """
    Loads the USUC dataset from given folder

    :param data_dir: Folder of USUC dataset
    :return: (time sequences, dataset config)
    """

    if len(os.listdir(data_dir)) == 0:
        generate_dataset(data_dir=data_dir)
    # load meta info
    with open(data_dir + CONFIG_FILEPATH) as f:
        config = json.load(f)

    with open(data_dir + TIME_SEQUENCES_FILEPATH, "rb") as f:
        windows = dill.load(f)

    return windows, config


def generate_dataset(
    num_actions=10,
    noise_offset=0.3,
    noisy_circular_sector=(0, math.pi),
    data_dir="../discrete-usuc-dataset",
    runs=900,
    time_steps=4,
):
    # creating empty dir (overwrites dir if it already exists)
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)

    # run generator
    env = usuc.USUCDiscreteEnv(
        num_actions, noisy_circular_sector, noise_offset, render=False
    )

    gen(env, runs, time_steps, data_dir)


if __name__ == "__main__":
    # env config
    num_actions = 10
    noise_offset = 0.3
    noisy_circular_sector = (0, math.pi)

    # generator config
    data_dir = "../discrete-usuc-dataset"
    runs = 900
    time_steps = 4

    # creating empty dir (overwrites dir if it already exists)
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)

    # run generator
    env = usuc.USUCDiscreteEnv(
        num_actions, noisy_circular_sector, noise_offset, render=False
    )

    gen(env, runs, time_steps, data_dir)
