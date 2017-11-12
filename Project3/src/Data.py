#!/usr/bin/env python3

#A class for the data sets, so we have an instance of the data and can reference different data sets from driver
#pass the number of classes
#call readData(fileName) to read the file
#call getFolds() to create the ten folds
#After that call, you can access the Data.crossValidatedTrain, Data.crossValidatedTrainOut
from sklearn.model_selection import KFold
import random
import numpy as np
#from src import helper
from . import helper
class Data:
    def __init__(self, numClasses):
        self.numberOfClasses = numClasses
        self.data = None
        self.expectedOut = None
        #will hold 10 folds, crossValidatedTrain[0] will be another matrix with the data points, so you can loop though it
        self.crossValidatedTrain = []
        self.crossValidatedTrainOut = []
        self.crossValidatedTest = []
        self.crossValidatedTestOut = []
        self.folds = 10

    def readData(self, fileName):
        self.data = []
        self.expectedOut = []
        f = open(fileName, 'r')
        for line in f:
            line = line.split('\t')
            if(line[len(line) - 1] == "\n"):
                line.pop()
            self.expectedOut.append(helper.toArrayRep(line.pop(),self.numberOfClasses))
            self.data.append(line)
        for k in self.data:
            for i in range(len(k)):
                k[i] = float(k[i])
        #map(float, self.data)
        #map(float, self.expectedOut)
        x = list(zip(self.data, self.expectedOut))
        random.shuffle(x)
        self.data, self.expectedOut = zip(*x)

    
    def getFolds(self):
        input = np.array(self.data)
        fold = KFold(n_splits= 10)
        count = 0
        for train,test in fold.split(input):
            self.crossValidatedTrain.append([])
            self.crossValidatedTest.append([])
            self.crossValidatedTestOut.append([])
            self.crossValidatedTrainOut.append([])
            for i in range(len(train)):
                self.crossValidatedTrain[count].append(self.data[train[i]])
                self.crossValidatedTrainOut[count].append(self.expectedOut[train[i]])
            for k in range(len(test)):
                self.crossValidatedTest[count].append(self.data[test[k]])
                self.crossValidatedTestOut[count].append(self.expectedOut[test[k]])
            count += 1
'''
d1 = Data(6)
d1.readData('test.txt')
print(d1.data)
print(d1.expectedOut)
d1.getFolds()
for i in range(len(d1.crossValidatedTrain)):
    print(d1.crossValidatedTrain[i],"trainIn" )
    print(d1.crossValidatedTrainOut[i], "trainOut")
    print(d1.crossValidatedTest[i], 'testin')
    print(d1.crossValidatedTestOut[i], "testOut")
'''
