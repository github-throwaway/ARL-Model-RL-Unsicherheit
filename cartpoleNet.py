import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import tensorflow.compat.v2 as tf
from swingup_wrapper import SwingUpWrapper
from neuralNetwork import neuralNetworkExpanded3, networkSample

keras = tf.keras
K = keras.backend
negloglik = lambda y, p_y: -p_y.log_prob(y)





class CartpoleNet():

    def __init__(self):
        print("init")

    def trainNeuralNetwork(self):
        file = open("outOfSample.csv")
        reader = csv.reader(file)

        my_data = genfromtxt('observations.csv', delimiter=',').astype(np.float32)
        # number of entries in rows -1
        data1, data2 = np.hsplit(my_data, [len(my_data[0]) - 1])

        myModel = neuralNetworkExpanded3(data1, data2)
        print("MODEL", myModel)

        return myModel


    def generateActions(self, numberOfsubDivision):
        actions = []

        #space between -1 and 1
        space = 2
        diff = space/numberOfsubDivision
        currentAction=-1
        for i in range(numberOfsubDivision-1):

            currentAction= currentAction+diff
            actions.append(currentAction)

        for i in range (len(actions)):
            if actions[i] < -1 or actions[i] > 1:
                actions.pop(i)

        return actions


    def predictAngle(self, actions, observations, myModel):

        predictedMean=[]
        predictedStd= []
        for i in range(len(actions)):
            currAction= actions[i]
            observations.append(currAction)

            x_tst = tf.expand_dims(observations, 0)

            yhat = myModel(x_tst)
            med = yhat.loc
            std = yhat.scale
            predictedMean.append(K.eval(np.squeeze(med)))
            predictedStd.append(K.eval(np.squeeze(std)))
            observations.pop(len(observations)-1)


        predictedMean = np.ravel(predictedMean)
        predictedStd = np.ravel(predictedStd)
        return predictedMean, predictedStd


    def calculateReward(self, angle, std):
       # TODO:calculate reward
        reward=0
        return reward


    def evaluateBestAction(self, predictedAngles, predictedStd, observations):
        rewards = []
        for i in range(len(predictedAngles)):
            reward= self.calculateReward(predictedAngles[i], predictedStd[i])
            rewards.append(reward)

        return rewards.index(max(rewards))











if __name__ == "__main__":
    env = SwingUpWrapper()
    cartNet = CartpoleNet()

    myModel = cartNet.trainNeuralNetwork()
    # myModel.save('models/medical_trial_model.h5')

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

                # print(obs)
                # print("action", action)

                observations.append(obs[0])
                observations.append(obs[1])
                observations.append(obs[4])
                # ONLY WHEN ACTION IS SAVED TOO
                observations.append(action[0])
                observations.append(obs[len(obs) - 1])

            elif len(observations) == numberOfValuesPerObservation * numberOfTimeSteps:
                #  take different actions and give it into neural network
                actions = cartNet.generateActions(100)
                predictedAngles, predictedStd = cartNet.predictAngle(actions, observations, myModel)
                action = actions[cartNet.evaluateBestAction(predictedAngles, predictedStd, observations)]
                #action = env.org_env.action_space.sample()


                # INTO NN:
                # observations.append(action[0])
                # x_tst = tf.expand_dims(observations, 0)
                # # med, std = networkSample(myModel, 1, x_tst)
                # yhat = myModel(x_tst)
                # med = yhat.loc
                # std = yhat.scale
                #
                # print(K.eval(np.squeeze(med)))
                # print(K.eval(np.squeeze(std)))

                # med = myModel.predict(x_tst)

                obs, rew, done, info = env.step(action)

                print("action", action)

                #observations.pop(len(observations) - 1)
                observations.append(obs[0])
                observations.append(obs[1])
                observations.append(obs[4])
                # ONLY WHEN ACTION IS SAVED TOO
                #observations.append(action[0])
                observations.append(action)
                observations.append(obs[len(obs) - 1])

                observations.pop(0)
                observations.pop(0)
                observations.pop(0)
                observations.pop(0)
                # ONLY WHEN ACTION IS SAVED TOO
                observations.pop(0)

                env.org_env.render()
