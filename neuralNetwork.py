import math
import pandas as pd
from collections import namedtuple
from random import uniform
from numpy import genfromtxt
import gym
import numpy as np


import tensorflow as tf
#todo for expanded
#import tensorflow.compat.v2 as tf

keras = tf.keras
K = keras.backend

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers

KL = keras.optimizers
KD = keras.Sequential
import tensorflow_probability as tfp
tfd = tfp.distributions

#todo for expanded
#from tensorflow_probability.python.layers.dense_variational_v2 import _make_kl_divergence_penalty



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
        tf.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1 / data1.shape[0]),
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


class DenseVariational(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseVariational, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            **kwargs)
        self.units = int(units)

        self._make_posterior_fn = make_posterior_fn
        self._make_prior_fn = make_prior_fn
        self._kl_divergence_fn = _make_kl_divergence_penalty(
            kl_use_exact, weight=kl_weight)

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = False
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `DenseVariational` '
                             'should be defined. Found `None`.')
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=2, axes={-1: last_dim})

        self._posterior = self._make_posterior_fn(
            last_dim * self.units,
            self.units if self.use_bias else 0,
            dtype)
        self._prior = self._make_prior_fn(
            last_dim * self.units,
            self.units if self.use_bias else 0,
            dtype)

        self.built = True

    def call(self, inputs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')

        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q)
        prev_units = self.input_spec.axes[-1]
        if self.use_bias:
            split_sizes = [prev_units * self.units, self.units]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
        else:
            kernel, bias = w, None

        kernel = tf.reshape(kernel, shape=tf.concat([
            tf.shape(kernel)[:-1],
            [prev_units, self.units],
        ], axis=0))
        outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return outputs


if __name__ == "__main__":


    #print("CSV An Stelle 12: ", csv_obs.values[12])
    # csv_obs.values[12][0] returns the index 12, so start with 1
    csv_obs = pd.read_csv('observations.csv')
    my_data = genfromtxt('observations.csv', delimiter=',').astype(np.float32)

    data1, data2 = np.hsplit(my_data, [9])
    #print(data1)
    #print(data2)

    neuralNetworkSimple()
    neuralNetworkStandardDev()

    #denseVar= DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1 / data1.shape[0]), tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    #neuralNetworkExpanded()






