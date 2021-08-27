import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import tensorflow.compat.v2 as tf
from swingup_wrapper import SwingUpWrapper
from neuralNetwork import neuralNetworkExpanded3, networkSample

negloglik = lambda y, p_y: -p_y.log_prob(y)


class CartpoleNet():

    def __init__(self):
        print("init")




    def trainNeuralNetwork(self):
        file = open("outOfSample.csv")
        reader = csv.reader(file)
        number_of_rows = len(list(reader))

        my_data = genfromtxt('observations.csv', delimiter=',').astype(np.float32)
        # number of entries in rows -1
        data1, data2 = np.hsplit(my_data, [len(my_data[0]) - 1])

        out_of_sample = genfromtxt('outOfSample.csv', delimiter=',').astype(np.float32)
        run1, run2 = np.hsplit(out_of_sample, [len(out_of_sample[0]) - 1])

        myModel = neuralNetworkExpanded3(data1, data2)
        print("MODEL", myModel)
        print("MODEL summary", myModel)

        # for index in range(1, number_of_rows, 1):
        #     # for index in range(1, 100, 1):
        #     # x_tst = tf.expand_dims(data1[index, :], 0)
        #     x_tst = tf.expand_dims(run1[index, :], 0)
        #     med, std = networkSample(myModel, 100, x_tst)
        #
        #     print(med)
        #     print(std)

        return myModel


if __name__ == "__main__":
    env = SwingUpWrapper()
    cartNet = CartpoleNet()

    myModel = cartNet.trainNeuralNetwork()
    #myModel.save('models/medical_trial_model.h5')

    observations = []
    numberOfValuesPerObservation = 5
    numberOfTimeSteps = 4

    for _ in range(1):

        done = False
        state = env.reset()

        while not done:
            if len(observations) < numberOfValuesPerObservation * numberOfTimeSteps:
                action = env.org_env.action_space.sample()
                obs, rew, done, info = env.step(action)

                print(obs)
                print("action", action)

                observations.append(obs[0])
                observations.append(obs[1])
                observations.append(obs[4])
                # ONLY WHEN ACTION IS SAVED TOO
                observations.append(action[0])
                observations.append(obs[len(obs) - 1])

            elif len(observations) == numberOfValuesPerObservation * numberOfTimeSteps:
                # TODO: take no random action, take different actions and give it into neural network
                action = env.org_env.action_space.sample()
                observations.append(action[0])

                # TODO: INTO NN:
                print("test obs", observations)
                x_tst = tf.expand_dims(observations, 0)
                med, std = networkSample(myModel, 100, x_tst)

                # TODO calculate which action is best and safest

                obs, rew, done, info = env.step(action)

                print(obs)
                print("action", action)

                observations.pop(len(observations)-1)
                observations.append(obs[0])
                observations.append(obs[1])
                observations.append(obs[4])
                # ONLY WHEN ACTION IS SAVED TOO
                observations.append(action[0])
                observations.append(obs[len(obs) - 1])

                observations.pop(0)
                observations.pop(0)
                observations.pop(0)
                observations.pop(0)
                # ONLY WHEN ACTION IS SAVED TOO
                observations.pop(0)

                env.org_env.render()
