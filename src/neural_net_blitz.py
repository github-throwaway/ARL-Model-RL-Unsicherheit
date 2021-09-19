import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from tqdm import tqdm
import neural_net

class CancelOut(nn.Module):
    '''
    CancelOut Layer

    x - an input data (vector, matrix, tensor)
    '''

    def __init__(self, input_dim, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(input_dim, requires_grad=True) + 4)

    def forward(self, x):
        return x * torch.sigmoid(self.weights.float())


@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cancelOut = CancelOut(input_dim)
        self.blinear = BayesianLinear(input_dim, output_dim)

    def forward(self, x):
        x = self.cancelOut(x)
        return self.blinear(x)


def evaluate_regression(regressor, X, y, samples=100, std_multiplier=2, render=False):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()

    if render:
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.set_title('Sine')
        ax1.plot(y.detach().numpy()[:, 0], color='red', label='Ground Truth')
        ax1.plot(means.detach().numpy()[:, 0], color='blue', label='Predicted')
        ax1.set_xlabel('episode')

        ax2.set_title('Cosine')
        ax2.plot(y.detach().numpy()[:, 1], color='red', label='Ground Truth')
        ax2.plot(means.detach().numpy()[:, 1], color='blue', label='Predicted')
        ax2.set_xlabel('episode')

        ax1.legend()
        ax2.legend()
        plt.show()

    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


def generate_model():
    x_train, y_train, x_test, y_test = neural_net.load_discrete_usuc()
    dataloader_train, dataloader_test = neural_net.dataloaders(x_train, y_train, x_test, y_test)

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

    plt.plot(losses, "x", label="losses")
    plt.show()

    evaluate_regression(
        regressor,
        x_test,
        y_test,
        samples=25,
        std_multiplier=3,
        render=True
    )

    torch.save(regressor, "../models/blitz50k.pt")

if __name__ == '__main__':
    generate_model()
