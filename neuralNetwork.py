import math
import pandas as pd
from collections import namedtuple
from random import uniform
from numpy import genfromtxt
import gym
import numpy as np


import tensorflow as tf
keras = tf.keras
K = keras.backend
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
KL = keras.optimizers
import tensorflow_probability as tfp
tfd = tfp.distributions


negloglik = lambda y, p_y: -p_y.log_prob(y)



def neuralNetworkSimple():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)
    model.fit(data1, data2, epochs=500, verbose=False)

    x_tst = tf.expand_dims(data1[1, :], 0)
    # Make predictions.
    yhat = model(x_tst)
    print(yhat.mean())


if __name__ == "__main__":


    #print("CSV An Stelle 12: ", csv_obs.values[12])
    # csv_obs.values[12][0] returns the index 12, so start with 1
    csv_obs = pd.read_csv('observations.csv')
    my_data = genfromtxt('observations.csv', delimiter=',').astype(np.float32)

    data1, data2 = np.hsplit(my_data, [35])

    neuralNetworkSimple()






