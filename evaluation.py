from typing import List
import math
import matplotlib.pyplot as plt


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


def plot(ground_truths, predictions, upper_border=2 * math.pi, lower_border=0):
    # TODO
    data_dir = "./results"
    pred, std_dev = np.hsplit(predictions, [1])
    # Make sure, pred and std_dev are 1-d Arrays
    # TODO Either use np.concatenate or make sure, array is passed with fitting dimensions
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