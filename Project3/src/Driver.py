from src.shared.forwardProp import ForwardProp
from src.Data import Data
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.GeneticAlg import GeneticAlg
from src.EvoStrategy import EvoStrat
from src.Tester import Tester
from src.backprop.bpAlg import BPAlg
from src.EvoAlg import EvoAlg

def testNetwork(network, testDataIns, testDataOuts):
    ea = EvoAlg(1, 2, 10, .1, testDataIns, testDataOuts)
    fitness = ea.evalFitness(network)   
    print("fitness", fitness)

    #Accepts a file name, will return the data, and expeceted output in an array [data,out]
#Sample data set
#Pass number of classes
dataSet1 = Data(7)
#Pass file name
dataSet1.readData('energyTestCropped.txt')
#do the foldin
dataSet1.getFolds()
trainingDataIn = dataSet1.crossValidatedTrain[0]
trainingDataOut = dataSet1.crossValidatedTrainOut[0]
testDataIn = dataSet1.crossValidatedTest[0]
testDataOut = dataSet1.crossValidatedTestOut[0]
hiddenLayers = 2
nodesPerHL = 10
#one set of data points, the first fold, we can train and test by iterating a loop over dataSet1.CrossValidatedTrain, etc
#to get the 10 folds
#print(dataSet1.crossValidatedTrain[0])
#print(dataSet1.crossValidatedTrainOut[0])
#GA = GeneticAlg(50, hiddenLayers, nodesPerHL, .1, trainingDataIn, trainingDataOut)
#GA.train(1)
#EVO = EvoStrat(50, hiddenLayers, 1nodesPerHL, .1, trainingDataIn, trainingDataOut)
#EVO.train(10)
# train backprop
backprop = BPAlg()
trainedNetwork = backprop.train(trainingDataIn, trainingDataOut, hiddenLayers, nodesPerHL)
#test backprop
testNetwork(trainedNetwork, testDataIn, testDataOut)

