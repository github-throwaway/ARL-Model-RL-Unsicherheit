import json
from typing import List
from uuid import uuid4

import shutil
import dill
import gym
from tqdm import tqdm

import usuc

# path constants
TIME_SEQUENCES_FILEPATH = "/time_sequences.p"
CONFIG_FILEPATH = "/config.json"


def gen_history(env, filename: str = None) -> List[tuple]:
    """
    Generates history for one run of a USUC env with random actions.
    Each time step is represented as a tuple (observation, info dict).

    :param env: Initialized env instance
    :param filename: Optional filename where to store observations
    :return: History of random run
    """

    history = usuc.random_actions(env)

    if filename:
        with open(filename, "wb") as f:
            dill.dump(history, f)

    return history


def gen_time_sequences(history: list, time_steps: int, filename: str = None) -> List[List[tuple]]:
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
        windows.append(history[i - time_steps: i + 1])

    # save time sequences if filename is provided
    if filename:
        with open(filename, 'wb') as f:
            dill.dump(windows, f)

    return windows


def gen():
    """"
    Generates USUC dataset
    """
    import os
    import math

    data_dir = "../discrete-usuc-dataset"
    runs = 1000
    num_actions = 100
    noise_offset = 0.5
    noisy_circular_sector = (0, math.pi)
    time_steps = 4
    env = usuc.USUCDiscreteEnv(num_actions, noisy_circular_sector, noise_offset,
                               render=False)

    # creating empty dir (overwrites dir if it already exists)
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(dir)

    # generationg dataset
    print("generating dataset...")
    windows = []
    for _ in tqdm(range(runs)):
        filename = data_dir + "/" + str(uuid4().time_low)
        env.reset(usuc.random_start_theta())
        history = gen_history(env, filename + "-rec.p")

        # generate time sequences from history of current run
        windows.extend(gen_time_sequences(history, time_steps))

    # save time sequences
    print(f"saving time sequences in {TIME_SEQUENCES_FILEPATH} (this may take a while)")
    with open(data_dir + TIME_SEQUENCES_FILEPATH, 'wb') as f:
        dill.dump(windows, f)

    # save meta info
    print(f"saving meta info in {CONFIG_FILEPATH}")
    with open(data_dir + CONFIG_FILEPATH, "w") as json_file:
        meta = {
            "num_time_sequences": len(windows),
            "num_actions": env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else None,
            "number_of_runs": runs,
            "time_steps": time_steps,
            "noisy_circular_sector": env.ncs,
            "noise_offset": env.noise_offset,
        }
        json.dump(meta, json_file)


def load(data_dir: str) -> List[tuple]:
    """
    Loads the USUC dataset from given folder

    :param data_dir: Folder of USUC dataset
    :return: (time sequences, dataset config)
    """

    # load meta info
    with open(data_dir + CONFIG_FILEPATH) as f:
        config = json.load(f)

    with open(data_dir + TIME_SEQUENCES_FILEPATH, "rb") as f:
        windows = dill.load(f)

    return windows, config


# if __name__ == '__main__':
#     gen()
