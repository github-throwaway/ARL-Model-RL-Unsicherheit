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
        cartNet= CartpoleNet()

        myModel = cartNet.trainNeuralNetwork()

        for _ in range(4):

            done = False
            state = env.reset()

            while not done:
                action = env.org_env.action_space.sample()

                obs, rew, done, info = env.step(action)

                env.org_env.render()

                print(obs)
                print("action", action)
                # print(info)
            # print("space", action_space)
