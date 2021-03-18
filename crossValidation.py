import numpy as np

# This script takes both our input matrix and our output vector and divides it into folds for cross validation.
# We start by generating a random seed that will determine the division of training / test data, randomly.
# The rest of the function is dedicated to making sure everything finds itself in the right place.

def Kfold_crossVal(inputs, outputs, F=10):
    seed = np.random.choice(len(inputs), len(inputs), replace=False)

    trainX, trainY, testX, testY = list(), list(), list(), list()

    split = len(inputs) / F

    for i in range(F):
        start = int(split * i)
        end = int(split * (i + 1))

        hold = np.zeros((len(seed), np.size(inputs, axis=1)))
        hold[:,0] = seed

        testX.append(hold[start : end])
        testY.append(seed[start : end])

        temp = seed
        temp = np.delete(temp, np.arange(start, end))

        hold = np.zeros((len(temp), np.size(inputs, axis=1)))
        hold[:,0] = temp

        trainX.append(hold)
        trainY.append(temp)

    for i in range(F):
        for j in range(0, len(trainX[i])):
            trainX[i][j] = inputs[int(trainX[i][j,0])]
            trainY[i][j] = outputs[int(trainY[i][j])]

        for j in range(0, len(testX[i])):
            testX[i][j] = inputs[int(testX[i][j,0])]
            testY[i][j] = outputs[int(testY[i][j])]

    return trainX, trainY, testX, testY