import math
import os
from typing import List

import numpy as np
import tikzplotlib
import torch
import utils
from data import gen
import matplotlib.pyplot as plt


def plot_sin_cos_with_stds(history, filename: str = None, create_tex=False):
    x_values = list(range(len(history)))

    observed_theta_sin = []
    observed_theta_cos = []

    pred_theta_sin = []
    pred_theta_cos = []

    std_sin = []
    std_cos = []

    bound_sin = []
    bound_cos = []

    for step in history:
        obs, _, _, info = step

        observed_theta_sin.append(info["observed_theta_sin"])
        observed_theta_cos.append(info["observed_theta_cos"])

        pred_theta_sin.append(obs.theta_sin)
        pred_theta_cos.append(obs.theta_cos)

        std_sin.append(info["predicted_std_sin"])
        std_cos.append(info["predicted_std_cos"])

        std_multiplier = 2
        bound_sin.append(
            (
                pred_theta_sin[-1] - (std_multiplier * std_sin[-1]),
                pred_theta_sin[-1] + (std_multiplier * std_sin[-1]),
            )
        )
        bound_cos.append(
            (
                pred_theta_cos[-1] - (std_multiplier * std_cos[-1]),
                pred_theta_cos[-1] + (std_multiplier * std_cos[-1]),
            )
        )

    fig, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    ### sine
    # Plot ground truth data as black stars
    y1_ax.plot(x_values, observed_theta_sin, "k*")

    # Predictive mean as blue line
    y1_ax.plot(x_values, pred_theta_sin, "b")

    # Shade in confidence
    y1_ax.fill_between(
        x_values,
        [lower for lower, _ in bound_sin],
        [upper for _, upper in bound_sin],
        alpha=0.5,
    )
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(["Sin", "Mean", "Confidence"])
    y1_ax.set_title("Sin Values (Likelihood)")

    ### cosine
    # Plot ground truth data as black stars
    y2_ax.plot(x_values, observed_theta_cos, "k*")

    # Predictive mean as blue line
    y2_ax.plot(x_values, pred_theta_cos, "b")

    # Shade in confidence
    y2_ax.fill_between(
        x_values,
        [lower for lower, _ in bound_cos],
        [upper for _, upper in bound_cos],
        alpha=0.5,
    )
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(["Cos", "Mean", "Confidence"])
    y2_ax.set_title("Cos Values (Likelihood)")

    if filename:
        plt.savefig(f"../plots/sin_cos_std_{filename}.png")
        if create_tex:
            tikzplotlib.save(f"../plots/sin_cos_std_{filename}.tex")
    else:
        plt.show()


def plot_reward_angle(history, filename: str = None, create_tex=False):
    angles = []
    rewards = []
    x_pos = []

    for step in history:
        (obs, reward, done, info) = step
        # angles.append(utils.calc_theta(obs.theta_sin, obs.theta_cos))
        # angles.append(utils.calc_theta(info["observed_theta_sin"],
        # info["observed_theta_cos"]))
        angles.append(info["original_theta"])
        x_pos.append(obs.x_pos)
        rewards.append(reward)

    plt.plot(angles, "x", label="Angles")
    plt.plot(x_pos, "x", label="x_pos")
    plt.plot(rewards, "x", label="Rewards")

    plt.legend()
    if filename:
        plt.savefig(f"../plots/reward_angle_{filename}.png")
        if create_tex:
            tikzplotlib.save(f"../plots/reward_angle_{filename}.tex")
    else:
        plt.show()


def plot_uncertainty(history, filename: str = None, create_tex=False):
    std = [(info["predicted_std_sin"], info["predicted_std_cos"]) for obs, reward, done, info in history]
    uncertainty = [math.sqrt(std_sin ** 2 + std_cos ** 2) for std_sin, std_cos in std]

    plt.plot(uncertainty, "x", label="Uncertainty")

    plt.legend()
    if filename:
        plt.savefig(f"../plots/reward_angle_{filename}.png")
        if create_tex:
            tikzplotlib.save(f"../plots/reward_angle_{filename}.tex")
    else:
        plt.show()


def plot_angles(
    history: List[tuple], model_name, filename: str = None, show=True, create_tex=False
) -> None:
    """
    Plots the original angles as well as the observed angle in one figure
    for comparison

    :param original: The original angles
    :param observed: The observed angles including noise (i.e. with
    uncertainty)
    :param tikz_filepath: Optional filepath where figure is saved as tikz
    """

    original = [
        (info["original_theta_sin"], info["original_theta_cos"])
        for (_, __, ___, info) in history
    ]
    observed = [(obs.theta_sin, obs.theta_cos) for (obs, _, __, ___) in history]

    fig = plt.figure(figsize=(19, 12))
    plt.title(f"Angle Progression with {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Pole Angle")

    plt.plot(
        [sin for sin, _ in original], "x", label="Original Angle Sin", color="blue"
    )
    plt.plot([sin for sin, _ in observed], "x", label="Observed Angle Sin", color="red")
    plt.plot(
        [cos for _, cos in original], "x", label="Original Angle Cos", color="green"
    )
    plt.plot(
        [cos for _, cos in observed], "x", label="Observed Angle Cos", color="orange"
    )

    plt.legend()
    plt.grid()

    if filename:
        plt.savefig(f"../plots/angles_{filename}.png")
        if create_tex:
            tikzplotlib.save(f"../plots/angles_{filename}.tex")

    if show:
        # if run from the command line this blocks the the code execution
        # until the windows is closed manually
        # todo: save figure instead  or used interactive mode?
        # https://stackoverflow.com/a/458295
        plt.show()

    plt.close(fig)


def plot(ground_truths, predictions, upper_border=2 * math.pi, lower_border=0):
    # TODO
    data_dir = "./results"
    pred, std_dev = np.hsplit(predictions, [1])
    # Make sure, pred and std_dev are 1-d Arrays
    # TODO Either use np.concatenate or make sure, array is passed with
    #  fitting dimensions
    pred = np.concatenate(pred, axis=None)
    std_dev = np.concatenate(std_dev, axis=None)
    fig = plt.figure(figsize=(19, 12))

    plt.plot(ground_truths, label="Ground Truths", color="blue")
    plt.plot(pred, label="Predictions", color="orange")
    """
    higher_dev = [min(p + s, upper_border) for p, s in zip(pred, std_dev)]
    lower_dev = [max(p - s, lower_border) for p, s in zip(pred, std_dev)]
    x = range(len(std_dev))
    plt.fill_between(x, higher_dev, lower_dev, color='yellow', alpha=0.3)
    plt.plot(higher_dev, color='yellow', alpha=0.5)
    plt.plot(lower_dev, color='yellow', alpha=0.5)
    """
    plt.xlabel("Time")
    plt.ylabel("Pole Angle")
    plt.legend()
    plt.grid()
    plt.savefig(data_dir + "/plot.png")
    plt.show()
    plt.close(fig)


def plot_predictions():
    # TODO
    data_dir = "./results"

    # load ground truths
    _, (_, y_test) = gen.load_datasets(data_dir)
    y_test = list(np.delete(y_test, 0, 1).flat)
    print(y_test)

    # load predictions
    predictions = np.genfromtxt(data_dir + "/predictions.csv", delimiter=",")

    # check if data is continuous
    for i in range(len(y_test) - 1):
        if abs(y_test[i] - y_test[i + 1]) > 0.5:
            print(i, y_test[i], y_test[i + 1])

    # plot
    plot(y_test, predictions)


def plot_summary(observation, info, filename="summary_plots"):
    """
    Plots a summary of the comparisons of sine, cosine, the angle and the
    rewards of the observed and the predicted data

    :param observation: The observations from the environment
    :param info: The observed data generated by our nn
    :param filename: The name of the created plot file
    """
    x_pos, x_dot, theta_sin, theta_cos, theta_dot = observation
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    ax1.set_title("Sine")
    ax1.plot(theta_sin, color="red", label="Observed Sine")
    ax1.plot(info["predicted_theta_sin"], color="blue", label="Predicted Sine")
    ax1.set_xlabel("episode")

    ax2.set_title("Cosine")
    ax2.plot(theta_cos, color="red", label="Observed Cosine")
    ax2.plot(info["predicted_theta_cos"], color="blue", label="Predicted Cosine")
    ax2.set_xlabel("episode")

    ax3.set_title("Angle")
    observed_angle = np.arctan2(theta_sin, theta_cos)
    predicted_angle = np.arctan2(
        info["predicted_theta_sin"], info["predicted_theta_cos"]
    )
    ax3.plot(observed_angle, color="red", label="Observed Angle")
    ax3.plot(predicted_angle, color="blue", label="Predicted Angle")
    ax3.set_xlabel("episode")

    ax4.set_title("Reward")
    rewards = info["rewards"]
    ax4.plot(rewards, color="red")
    ax4.set_xlabel("episode")

    plot_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/plots/" + filename + ".png"
    )
    plt.savefig(plot_path)
    plt.show()


def plot_test(
    original: List[float],
    observed: List[float],
    predicted: List[float],
    std: List[float],
    reward: List[float],
    filepath: str = None,
    show=True,
) -> None:
    """
    Plots the original angles as well as the observed angle in one figure
    for comparison

    :param original: The original angles
    :param observed: The observed angles including noise (i.e. with
    uncertainty)
    :param filepath: Optional filepath where figure is saved
    """

    assert len(original) == len(observed), "Length of the lists do not match"

    fig = plt.figure(figsize=(19, 12))
    plt.title("Angle Progression")
    plt.xlabel("Time")
    plt.ylabel("Pole Angle")

    # plt.plot(original, 'x', label='Original Angle', color="blue")
    plt.plot(observed, "x", label="Observed Angle", color="orange")
    plt.plot(predicted, "x", label="Predicted Angle", color="green")
    plt.plot(std, "x", label="Std", color="red")
    plt.plot(reward, "x", label="Reward", color="yellow")

    plt.legend()
    plt.grid()

    if show:
        plt.show()

    if filepath:
        plt.savefig(filepath)

    plt.close(fig)


def plot_general_summary(histories, nn_model, agent_name):
    """
    This function is used to call the four basic plotting functions,
    needed to evaluate the model and the agent

    :param histories: The histories of the run
    :param nn_model: The name of the used nn_model
    :param agent_name: The name of the agent to determine the target files
    """
    len_history = [len(h) for h in histories]
    max_history_index = len_history.index(max(len_history))

    plot_uncertainty(histories[max_history_index], filename=agent_name, create_tex=True)
    plot_angles(histories[max_history_index], nn_model, filename=agent_name, create_tex=True)
    plot_reward_angle(histories[max_history_index], filename=agent_name, create_tex=True)
    plot_sin_cos_with_stds(histories[max_history_index], filename=agent_name, create_tex=True)


# TODO
def evaluate_regression(regressor, x, y, samples=25, std_multiplier=2, render=False):
    preds = [regressor(x) for _ in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)

    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()

    print("CI acc:", {ic_acc}),
    print("CI upper acc:", (ci_upper >= y).float().mean()),
    print("CI lower acc:", (ci_lower <= y).float().mean())

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_title("Sine")
    ax1.plot(y.detach().numpy()[:, 0], color="red", label="Ground Truth")
    ax1.plot(means.detach().numpy()[:, 0], color="blue", label="Predicted")
    ax1.set_xlabel("episode")

    ax2.set_title("Cosine")
    ax2.plot(y.detach().numpy()[:, 1], color="red", label="Ground Truth")
    ax2.plot(means.detach().numpy()[:, 1], color="blue", label="Predicted")
    ax2.set_xlabel("episode")

    ax1.legend()
    ax2.legend()
    plt.show()


def plot_losses(losses):
    print("Final Loss:", losses[-1])

    fig = plt.figure(figsize=(19, 12))

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.plot(losses, "*", label="Losses")
    plt.legend()
    plt.show()
