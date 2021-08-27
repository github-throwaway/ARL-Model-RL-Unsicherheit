import csv
import math
import statistics
import matplotlib.pyplot as plt

import pandas as pd
from collections import namedtuple
from random import uniform
from numpy import genfromtxt
import gym
import numpy as np

# import tensorflow as tf
import tensorflow.compat.v2 as tf
import pickle
import os

keras = tf.keras
K = keras.backend

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers

KL = keras.optimizers
KD = keras.Sequential
import tensorflow_probability as tfp

tfd = tfp.distributions


from tensorflow_probability.python.layers.dense_variational_v2 import _make_kl_divergence_penalty, DenseVariational


negloglik = lambda y, p_y: -p_y.log_prob(y)



def neuralNetworkSimple():
    # No Uncertainty
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)
    model.fit(data1, data2, epochs=500, verbose=False)


    x_tst = tf.expand_dims(data1[1, :], 0)
    # print(x_tst)
    print(K.eval(x_tst))
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


def neuralNetworkStandardDev():
    # Aleatoric Uncertainty
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

    # x_tst = tf.expand_dims(data1[1, :], 0)
    # print(x_tst)
    # print(K.eval(x_tst))
    # Make predictions.

    # print(model.get_weights())
    # print("------")
    # print(model.get_weights()[0])
    # print(model.get_weights()[1])
    # yhat = model(x_tst)
    # print(K.eval(yhat.mean()))
    # print(K.eval(yhat.variance()))
    # print("------")
    # assert isinstance(yhat, tfd.Distribution)
    # test_mult=tf.math.multiply(tf.transpose(x_tst),model.weights[0])
    # print(K.eval(test_mult))

    [print(np.squeeze(w.numpy())) for w in model.weights];
    yhat = model(x_tst)
    assert isinstance(yhat, tfd.Distribution)


def neuralNetworkExpanded2():
    # Epistemic Uncertainty
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1 / data1.shape[0]),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=negloglik)
    model.fit(data1, data2, epochs=1000, verbose=False);

    # print(x_tst)
    # print(K.eval(x_tst))
    # Make predictions.

    # print(model.get_weights())
    # print("------")
    # print(model.get_weights()[0])
    # print(model.get_weights()[1])
    # yhat = model(x_tst)
    # yhats = [model(x_tst) for _ in range(3)]
    # print(K.eval(yhats))
    # print(K.eval(np.squeeze(yhat.mean())))
    # print(K.eval(yhat.mean()))
    # print("------")
    # print(K.eval(yhat))
    # print(K.eval(yhat.variance()))
    # yhat = model(x_tst)
    # print(K.eval(np.squeeze(yhat.mean())))
    # print(K.eval(yhat.mean()))
    # print("------")
    # print(K.eval(yhat))
    # print(K.eval(yhat.variance()))

    # [print(np.squeeze(w.numpy())) for w in model.weights];
    # yhat = model(x_tst)
    # assert isinstance(yhat, tfd.Distribution)
    # TODO Find out how to serialize model
    # model.save('model.dat')
    return model

def neuralNetworkExpanded3(data1, data2):
    # Epistemic Uncertainty
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1 / data1.shape[0]),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=negloglik)
    model.fit(data1, data2, epochs=1000, verbose=False);


    return model


def networkSample(mymodel, size, x_tst):
    yhats = [mymodel(x_tst) for _ in range(size)]
    m = np.zeros(size)
    s = np.zeros(size)
    for i, yhat in enumerate(yhats):
        m[i] = np.squeeze(yhat.mean())
        # s[i] = np.squeeze(yhat.stddev())
    med = np.mean(m)
    # position = np.where(m == med)
    mystd = np.std(m)

    return med, mystd


def plot(ground_truths, predictions):
    # TODO Fix shapes of input etc
    # for k in range(len(ground_truths)):
    #     if ground_truths[k]>math.pi:
    #         ground_truths[k]=2*math.pi-ground_truths[k]
    #     if predictions[k]>math.pi:
    #         predictions[k]=2*math.pi-predictions[k]

    x_vals = np.arange(0, len(ground_truths)-1)
    #print("X_Vals: ", x_vals, " Länge: ", len(x_vals))
    fig = plt.figure(figsize=(19, 12))
    plt.plot(ground_truths, label='Truth')
    pred = [i[0] for i in predictions]
    pred_plot = plt.plot(pred, label='Predicitons', color='orange')
    std_dev = [i[1] for i in predictions]
    #print("Std_Dev: ", std_dev, " Länge: ", len(std_dev))
    higher_dev = np.zeros_like(std_dev)
    lower_dev = np.zeros_like(std_dev)
    for x in range(len(std_dev)):
        higher_dev[x] = min(pred[x] + std_dev[x], 1.5)
        lower_dev[x] = max(pred[x] - std_dev[x], -1.5)
    plt.fill_between(x_vals, higher_dev, lower_dev, color='yellow', alpha=0.3)
    plt.plot(higher_dev, color='yellow', alpha=0.5)
    plt.plot(lower_dev, color='yellow', alpha=0.5)
    plt.xlabel("Frames")
    plt.ylabel("Pole Angle")
    plt.legend()
    plt.grid()
    plt.savefig('plot.png')
    plt.show()
    plt.close(fig)


if __name__ == "__main__":

    # print("CSV An Stelle 12: ", csv_obs.values[12])
    # csv_obs.values[12][0] returns the index 12, so start with 1
    file = open("outOfSample.csv")
    reader = csv.reader(file)
    number_of_rows = len(list(reader))

    print(number_of_rows)
    my_data = genfromtxt('observations.csv', delimiter=',').astype(np.float32)
    # number of entries in rows -1
    #data1, data2 = np.hsplit(my_data, [17])
    lenlist= len(my_data[0])
    data1, data2 = np.hsplit(my_data, [len(my_data[0])-1])

    print(data1)
    print(data2)


    out_of_sample = genfromtxt('outOfSample.csv', delimiter=',').astype(np.float32)
    #run1, run2 = np.hsplit(out_of_sample, [17])
    run1, run2 = np.hsplit(out_of_sample, [len(out_of_sample[0])-1])

    print(run1)
    print(run2)

    # print(data1)
    # print(data2)
    # maxData2 = max(data2)
    # print("max", maxData2)
    # minData2 = min(data2)
    # print("min", minData2)

    # neuralNetworkSimple()
    # neuralNetworkStandardDev()
    # neuralNetworkExpanded()
    myModel = neuralNetworkExpanded2()
    print(myModel)
    predicted_angle = []
    for index in range(1, number_of_rows, 1):
        # for index in range(1, 100, 1):
        # x_tst = tf.expand_dims(data1[index, :], 0)
        x_tst = tf.expand_dims(run1[index, :], 0)
        med, std = networkSample(myModel, 100, x_tst)
        predicted_angle.append((med, std))

        #print(index, "/", number_of_rows, " ---------")
        print(med)
        print(std)

    # df = pd.DataFrame(predicted_angle)
    # df.to_csv("predictions.csv", index=None, header=None, mode="a")
    # plot(run2, predicted_angle)
