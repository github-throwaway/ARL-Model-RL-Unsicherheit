import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from numpy import genfromtxt

tfd = tfp.distributions
keras = tf.keras
backend = keras.backend
optimiziers = keras.optimizers

Sequential = keras.Sequential
VariableLayer = tfp.layers.VariableLayer
DistributionLambda = tfp.layers.DistributionLambda

# negative lok likelihood
nll = lambda y, p_y: -p_y.log_prob(y)


class DenseVariational(tfp.layers.DenseVariational):
    def get_config(self):
        config = super().get_config().copy()
        config["name"] = "custom_dense_variational"
        print("config:", config)
        return config


class NeuralNet():
    def __init__(self, model):
        self.model = model

    def predict(self, time_series: list) -> tuple:
        """
        Predicts next angle and std for a given time series
        :param time_series: Time series (Several observations t0, ..., tn)
        :return: predicted angle, predicted std
        """
        # magic
        # TODO: was passiert hier?
        x = tf.expand_dims(time_series, 0)

        # make predictions
        # don't use model.predict here, does not return std
        yhats = self.model(x)
        med = yhats.loc
        std = yhats.scale

        # TODO: log predictions

        # extract values from tensors
        med = backend.eval(np.squeeze(med))
        std = backend.eval(np.squeeze(std))

        # med = predicted angle, std = predicted std
        return med, std


def build(x_shape):
    """Builds model architecture
    :param x_shape: TODO document
    :return: model architecture
    """

    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return Sequential([
            VariableLayer(2 * n, dtype=dtype),
            DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return Sequential([
            VariableLayer(n, dtype=dtype),
            DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])

    model = Sequential([
        # TODO: was ist x_shape = x_train.shape[0] <- was macht das und welche größe sollte das sein
        DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1 / x_shape),
        DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
    ])

    return model


def train(model, x_train, y_train):
    """
    Train model with given training set
    :param model: Untrained model (i.e. model architecture)
    :param x_train: Input values
    :param y_train: Expected values
    :return: Trained model
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=nll)
    model.fit(x_train, y_train, epochs=1000, verbose=False)

    return model


def save(filepath, model):
    # TODO
    model.save(filepath)


def load():
    # TODO: move to if main... when model savings works
    # TODO: replace with data from data generator and transform data to fit input requirements
    training_set = genfromtxt('time_series.csv', delimiter=',').astype(np.float32)
    # TODO: check if len() can be replaced with training_set.shape
    x_train, y_train = np.hsplit(training_set, [len(training_set[0]) - 1])

    # TODO: shape[0]?
    model = build(x_train.shape[0])
    model = train(model, x_train, y_train)

    return NeuralNet(model)
