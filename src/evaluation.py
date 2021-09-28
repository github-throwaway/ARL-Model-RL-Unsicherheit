import math
from typing import List

import matplotlib.pyplot as plt
import tikzplotlib
import torch

import utils


def plot_bounds(history, filepath: str = None, create_tex=False):
    """
    Plots the bounds for sine and cosine and their respective std

    :param history: History of a run
    :param filepath: Filepath where plot is saved
    :param create_tex: Whether to create an additional .tex file
    """
    x_values = list(range(len(history)))

    observed_theta_sin, observed_theta_cos = [], []
    pred_theta_sin, pred_theta_cos = [], []
    std_sin, std_cos = [], []
    bound_sin, bound_cos = [], []

    # collect data points & calculate boundaries regarding std
    for step in history:
        obs, _, _, info = step

        observed_theta_sin.append(info["observed_theta_sin"])
        observed_theta_cos.append(info["observed_theta_cos"])

        pred_theta_sin.append(obs.theta_sin)
        pred_theta_cos.append(obs.theta_cos)

        std_sin.append(info["predicted_std_sin"])
        std_cos.append(info["predicted_std_cos"])

        std_multiplier = 6
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

    # plot data
    fig, (y1_ax, y2_ax) = plt.subplots(2, 1, figsize=(19, 12))

    # 1. Sine
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

    # 2. Cosine
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

    if filepath:
        plt.savefig(filepath + ".png")
        if create_tex:
            tikzplotlib.save(filepath + ".tex")
    else:
        plt.show()


def plot_rewards(history, filepath: str = None, create_tex=False):
    """
    Plots the reward as well as the angle and the position for each time step of a run

    :param history: History of a run
    :param filepath: Filepath where plot is saved
    :param create_tex: Whether to create an additional .tex file
    """

    angles, rewards, x_pos = [], [], []

    for step in history:
        (obs, reward, done, info) = step
        angles.append(utils.calc_theta(obs.theta_sin, obs.theta_cos))
        x_pos.append(obs.x_pos)
        rewards.append(reward)

    plt.plot(angles, "x", label="Angles")
    plt.plot(x_pos, "x", label="x_pos")
    plt.plot(rewards, "x", label="Rewards")

    plt.legend()
    if filepath:
        plt.savefig(filepath + ".png")
        if create_tex:
            tikzplotlib.save(filepath + ".tex")
    else:
        plt.show()


def plot_uncertainty(history, filepath: str = None, create_tex=False):
    """
    Plots the accumulated uncertainty of sine and cosine:
    sqrt(std_sin² + std_cos2²)

    :param history: History of a run
    :param filepath: Filepath where plot is saved
    :param create_tex: Whether to create an additional .tex file
    """
    std = [
        (info["predicted_std_sin"], info["predicted_std_cos"])
        for obs, reward, done, info in history
    ]
    uncertainty = [math.sqrt(std_sin ** 2 + std_cos ** 2) for std_sin, std_cos in std]

    plt.plot([std_cos for _, std_cos in std], "x", label="Uncertainty")

    plt.legend()
    if filepath:
        plt.savefig(filepath + ".png")
        if create_tex:
            tikzplotlib.save(filepath + ".tex")
    else:
        plt.show()


def plot_angles(history: List[tuple], filepath: str = None, create_tex=False):
    """
    Plots the original angles as well as the observed angle in one figure
    for comparison

    :param history: History of a run
    :param filepath: Filepath where plot is saved
    :param create_tex: Whether to create an additional .tex file
    """

    original = [
        (info["original_theta_sin"], info["original_theta_cos"])
        for (_, __, ___, info) in history
    ]
    obs = [
        (info["observed_theta_sin"], info["observed_theta_cos"])
        for (_, __, ___, info) in history
    ]
    pred = [(obs.theta_sin, obs.theta_cos) for (obs, _, __, ___) in history]

    fig, (sp1, sp2) = plt.subplots(2, 1, figsize=(19, 12))

    plt.title(f"Angle Progression")
    plt.xlabel("Time")
    plt.ylabel("Pole Angle")

    # sine
    sp1.plot([sin for sin, _ in original], label="Original Angle Sin", color="blue")
    sp1.plot([sin for sin, _ in obs], "x", label="Observed Angle Sin", color="red")
    sp1.plot([sin for sin, _ in pred], label="Predicted Angle Sin", color="black")

    # cosine
    sp2.plot([cos for _, cos in original], label="Original Angle Cos", color="green")
    sp2.plot([cos for _, cos in obs], "x", label="Observed Angle Cos", color="orange")
    sp2.plot([cos for _, cos in pred], label="Predicted Angle Cos", color="yellow")

    plt.legend()
    plt.grid()

    if filepath:
        plt.savefig(filepath + ".png")
        if create_tex:
            tikzplotlib.save(filepath + ".tex")
    else:
        plt.show()


def plot_summary(histories, neural_net, agent_name):
    """
    This function is used to call the four basic plotting functions,
    needed to evaluate the model and the agent.

    From the given histories only plots for the longest history will be created

    :param histories: The histories of the `n` runs
    :param neural_net: The name of the neural net
    :param agent_name: The name of the agent. Used for filenames
    """

    # find longest history
    len_history = [len(h) for h in histories]
    max_history_index = len_history.index(max(len_history))

    # plot data
    plot_uncertainty(
        histories[max_history_index],
        filepath=f"../plots/{neural_net}-{agent_name}_uncertainty",
        create_tex=True,
    )
    plot_angles(
        histories[max_history_index],
        filepath=f"../plots/{neural_net}-{agent_name}_angles",
        create_tex=True,
    )
    plot_rewards(
        histories[max_history_index],
        filepath=f"../plots/{neural_net}-{agent_name}_rewards",
        create_tex=True,
    )
    plot_bounds(
        histories[max_history_index],
        filepath=f"../plots/{neural_net}-{agent_name}_bounds",
        create_tex=True,
    )


def evaluate_neural_net(nn, x, y, samples=25, std_multiplier=2):
    """
    Plots ground truths and predictions of the given neural net for sine and cosine

    :param nn: Neural net
    :param x: Inputs
    :param y: Outputs/Labels
    :param samples: Number of samples to draw from neural net
    :param std_multiplier: Multiplier for std
    """

    preds = [nn(x) for _ in range(samples)]
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
    """
    Plots the loss for a training run

    :param losses: Losses over time
    """

    print("Final Loss:", losses[-1])

    fig = plt.figure(figsize=(19, 12))

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.plot(losses, "*", label="Losses")
    plt.legend()
    plt.show()
