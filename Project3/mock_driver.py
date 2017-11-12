#!/usr/bin/env python3

from src.Data import Data
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.DifferentialEvo import DifferentialEvolution as DE
from src.Tester import Tester as T
import sys

# sys argv[1] - Size of population
# sys argv[2] - # of generations

data = Data(7)
#data.readData('src/indians2.txt')
#data.readData('src/energyTestCropped.txt')
#data.readData('src/shuttle.txt')
data.readData('src/shuttle.trim.txt')
data.getFolds()
de = DE(int(sys.argv[1]), 2, 10, None, data.crossValidatedTrain[0], data.crossValidatedTrainOut[0])
bestNetwork, bestFitness, bestIndex = de.train(int(sys.argv[2]), 0.47, 0.88) # 0.47, 0.88
#print(bestFitness)
