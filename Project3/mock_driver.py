#!/usr/bin/env python3

from src.Data import Data
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.DifferentialEvo import DifferentialEvolution as DE
from src.Tester import Tester as T

data = Data(7)
data.readData('src/energyTestCropped.txt')
data.getFolds()
de = DE(75, 2, 10, None, data.crossValidatedTrain[0], data.crossValidatedTrainOut[0])
bestNetwork, bestFitness = de.train(100, 0.47, 0.88)
print(bestFitness)
