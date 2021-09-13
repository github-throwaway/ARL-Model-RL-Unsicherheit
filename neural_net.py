import json
import os

# disable tf verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import data
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from numpy import genfromtxt
from itertools import chain

tfd = tfp.distributions
keras = tf.keras
backend = keras.backend
optimizers = keras.optimizers

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

Sequential = keras.Sequential
VariableLayer = tfp.layers.VariableLayer
DistributionLambda = tfp.layers.DistributionLambda

# negative log likelihood
nll = lambda y, p_y: -p_y.log_prob(y)

META_INFO_FILEPATH = "/meta.json"
CHECKPOINT_PATH = "/checkpoint"


class DenseVariational(tfp.layers.DenseVariational):
    def get_config(self):
        config = super().get_config().copy()
        config["name"] = "custom_dense_variational"
        print("config:", config)
        return config


class NeuralNet:
    def __init__(self, model, time_steps):
        """

        :param model:
        :param time_steps: Number of time steps for input for prediction
        """
        assert time_steps > 0, "Param time_steps must be greater than 0"

        self.model = model
        self.time_steps = time_steps

    def predict(self, recent_history: list, action) -> tuple:
        """
        Predicts next angle and std for a given time series
        # TODO: reference time_steps set in in init of this NN (used by usuc with NN)
        :param recent_history: The n recent observations t0, ..., tn)
        :param action: The current action to transition from tn to tn+1
        :return: predicted angle, predicted std
        """

        # transform recent history and current action to valid input for nn
        time_series = _transform(recent_history, action)

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


def _transform(recent_history, current_action):
    """
    # TODO: change doc
    Reorders values from observation and the action in list to feed neural network with (needs special order)
    :param observation:
    :param current_action:
    :return: Reordered values
    """

    # TODO: replace with dict calls (everywhere where obs is destructured)
    # TODO: check if this works correctly
    # TODO: make destructuring absolute without errors! (maybe assert or smth) -> or give this to neural net which destructures it
    # reorder values
    # reorder value function (input must be in special order for nn)
    reorder = lambda obs, action: [
        obs.x_pos,
        obs.x_dot,
        obs.theta_dot,
        action,
        obs.theta,
    ]
    time_series = [reorder(obs, info["action"]) for (obs, info) in recent_history]

    # flatten
    time_series = list(chain.from_iterable(time_series))

    # append current action
    time_series.append(current_action)

    return time_series


def build(rows) -> keras.Model:
    """Builds model architecture
    :param rows: TODO document
    :return: model architecture
    """

    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.0))
        return Sequential(
            [
                VariableLayer(2 * n, dtype=dtype),
                DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(
                            loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                ),
            ]
        )

    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return Sequential(
            [
                VariableLayer(n, dtype=dtype),
                DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                    )
                ),
            ]
        )

    model = Sequential(
        [
            # TODO: was ist x_shape = x_train.shape[0] <- was macht das und welche größe sollte das sein
            DenseVariational(
                1 + 1, posterior_mean_field, prior_trainable, kl_weight=1 / rows,
                input_shape=(21,)
            ),
            DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])
                )
            ),
        ]
    )

    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=nll)
    return model


def train(model_dir, model, x_train, y_train, x_test, y_test):
    """
    Train model with given training set
    :param model_dir: todo
    :param model: Untrained model (i.e. model architecture)
    :param x_train: Input values
    :param y_train: Expected values
    :param x_test:
    :param y_test:
    :return: Trained model
    """
    checkpoint_path = model_dir + CHECKPOINT_PATH

    # callback that saves the model's weights
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    # train model with callback
    model.fit(
        x_train,
        y_train,
        epochs=1000,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback],
    )

    return model


def load(model_dir="./model"):
    """
    todo: document
    :param model_dir:
    :return:
    """

    # load meta info
    with open(model_dir + META_INFO_FILEPATH) as json_file:
        config = json.load(json_file)
        print("Using config:", config)

    # build model
    model = build(config["rows"])

    # load latest checkpoint
    latest = tf.train.latest_checkpoint(model_dir)
    model.load_weights(latest).expect_partial()

    # init net
    net = NeuralNet(model, config["time_steps"])
    return net


def sample(model, size, x_tst):
    yhats = model(x_tst)
    med = yhats.loc
    std = yhats.scale

    return med, std


def test():
    model = load().model

    testset = genfromtxt("testset.csv", delimiter=",").astype(np.float32)
    run, y_test = np.hsplit(testset, [len(testset[0]) - 1])

    predicted_angle = []
    rows = len(testset)
    for index in range(rows):
        x_tst = tf.expand_dims(run[index, :], 0)

        med, std = sample(model, 15, x_tst)
        predicted_angle.append((med, std))

        # print(index, "/", number_of_rows, " ---------")
        print("mean", med)
        print("std", std)


def evaluate(model, x_eval, y_eval):
    # Re-evaluate the model
    loss = model.evaluate(x_eval, y_eval, verbose=2)


def check_tensorflow():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto()

    if tf.config.list_physical_devices("GPU"):
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.visible_device_list = "1"

    tf.compat.v1.Session(config=config)


if __name__ == "__main__":
    print("checking tensorflow installation...")
    check_tensorflow()

    print("Python Version:", sys.version.replace("\n", ""))
    print("Tensorflow Version:", tf.version.VERSION)

    print("building and training network...")
    model_dir = "model"
    dataset_dir = "./discrete-usuc-dataset"
    time_sequences, config = data.load(dataset_dir)

    # transform data to fit input requirements
    # t0_xpos, t0_xdot, t0_thetadot, t0_action, t0_theta, t1_xpos, t1_xdot, t1_thetadot, t1_action, t1_theta,
    # t2_xpos, t2_xdot, t2_thetadot, t2_action, t2_theta, t3_xpos, t3_xdot, t3_thetadot, t3_action, t3_theta,
    # t4_action, t4_theta (t4_action = action to get from t3 to t4, t4_theta = predicted theta/otuput/y)

    transformed_dataset = []
    for ts in time_sequences:
        output_idx = len(ts) - 1
        output = ts[output_idx]
        inputs = ts[:output_idx]
        _, info = ts[output_idx]

        last_action = output[1]["action"]
        tts = _transform(inputs, last_action)

        # add output angle
        output_angle = output[0].theta
        tts.append(output_angle)

        # append transformed
        transformed_dataset.append(tts)

    transformed_dataset = np.array(transformed_dataset)

    # split dataset into training, evaluation and test sets
    # 70% = training set, 20% = evaluation set, 10% = test set
    size = len(transformed_dataset)
    split1 = int(size * 0.7)
    split2 = int(size * (0.7 + 0.2))
    train_dataset = transformed_dataset[:split1]
    eval_dataset = transformed_dataset[split1: split2]
    test_dataset = transformed_dataset[split2:]

    # split dataset into input and output matrix
    (x_train, y_train) = np.hsplit(train_dataset, [train_dataset.shape[1] - 1])
    (x_eval, y_eval) = np.hsplit(eval_dataset, [eval_dataset.shape[1] - 1])
    (x_test, y_test) = np.hsplit(test_dataset, [test_dataset.shape[1] - 1])

    model = build(x_train.shape[0])
    model.summary()
    train(model_dir, model, x_train, y_train, x_eval, y_eval)

    # save meta info
    with open(model_dir + META_INFO_FILEPATH, "w") as json_file:
        meta = {
            "rows": x_train.shape[0],
            "time_steps": config["time_steps"]
        }
        json.dump(meta, json_file)
