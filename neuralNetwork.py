import math
import pandas as pd
from collections import namedtuple
from random import uniform
from numpy import genfromtxt
import gym
import numpy as np


# import tensorflow as tf
# todo for expanded
import tensorflow.compat.v1 as tf

keras = tf.keras
K = keras.backend

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers

KL = keras.optimizers
KD = keras.Sequential
import tensorflow_probability as tfp
tfd = tfp.distributions

#todo for expanded
from tensorflow_probability.python.layers.dense_variational_v2 import _make_kl_divergence_penalty, DenseVariational



negloglik = lambda y, p_y: -p_y.log_prob(y)



def neuralNetworkSimple():
    #No Uncertainty
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)
    model.fit(data1, data2, epochs=500, verbose=False)

    x_tst = tf.expand_dims(data1[1, :], 0)
    #print(x_tst)
    #print(K.eval(x_tst))
    # Make predictions.

    #print(model.get_weights())
    #print("------")
    #print(model.get_weights()[0])
    #print(model.get_weights()[1])
    yhat = model(x_tst)
    print(K.eval(yhat.mean()))
    print(K.eval(yhat.variance()))
    #print("------")
    #assert isinstance(yhat, tfd.Distribution)
    #test_mult=tf.math.multiply(tf.transpose(x_tst),model.weights[0])
    #print(K.eval(test_mult))


def neuralNetworkStandardDev():
    #Aleatoric Uncertainty
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1 + 1),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)
    model.fit(data1, data2, epochs=500, verbose=False)

    x_tst = tf.expand_dims(data1[1, :], 0)
    #print(x_tst)
    #print(K.eval(x_tst))
    # Make predictions.

    #print(model.get_weights())
    #print("------")
    #print(model.get_weights()[0])
    #print(model.get_weights()[1])
    yhat = model(x_tst)
    print(K.eval(yhat.mean()))
    print(K.eval(yhat.variance()))
    #print("------")
    #assert isinstance(yhat, tfd.Distribution)
    #test_mult=tf.math.multiply(tf.transpose(x_tst),model.weights[0])
    #print(K.eval(test_mult))

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def neuralNetworkExpanded():
    # Epistemic Uncertainty
    model = tf.keras.Sequential([
        DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1 / data1.shape[0]),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)
    model.fit(data1, data2, epochs=500, verbose=False)

    x_tst = tf.expand_dims(data1[1, :], 0)
    # print(x_tst)
    # print(K.eval(x_tst))
    # Make predictions.

    # print(model.get_weights())
    # print("------")
    # print(model.get_weights()[0])
    # print(model.get_weights()[1])
    yhat = model(x_tst)
    print(K.eval(yhat.mean()))
    print(K.eval(yhat.variance()))
    # print("------")
    # assert isinstance(yhat, tfd.Distribution)
    # test_mult=tf.math.multiply(tf.transpose(x_tst),model.weights[0])
    # print(K.eval(test_mult))


if __name__ == "__main__":


    #print("CSV An Stelle 12: ", csv_obs.values[12])
    # csv_obs.values[12][0] returns the index 12, so start with 1
    csv_obs = pd.read_csv('observations.csv')
    my_data = genfromtxt('observations.csv', delimiter=',').astype(np.float32)

    data1, data2 = np.hsplit(my_data, [9])
    #print(data1)
    #print(data2)

    #neuralNetworkSimple()
    #neuralNetworkStandardDev()

    # denseVar= DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1 / data1.shape[0]), tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    neuralNetworkExpanded()






