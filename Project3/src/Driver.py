from src.shared.forwardProp import ForwardProp
from src.Data import Data
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.GeneticAlg import GeneticAlg
from src.Tester import Tester

class Driver:

    def __init__(self):    
        self.numberOfClasses = 3 # will need to change this so it's not hard coded
        self.data
        self.expectedOut
        self.gABest = []
        self.evoAlgBest = []
        self.evoStraBest = []

    def test(self, input, expectedOut, network):
        correct = 0 #incorrect by default
        fp = ForwardProp(network,input,expectedOut)
        # if classification is correct, return 1
        if (fp.getHypothesis() == expectedOut):
            correct = 1
        return correct
    #Accepts and array of percentages correct for two different training methods


    #Accepts a file name, will return the data, and expeceted output in an array [data,out]
#Sample data set
#Pass number of classes
dataSet1 = Data(7)
#Pass file name
dataSet1.readData('energy.txt')
#do the foldin
dataSet1.getFolds()
#one set of data points, the first fold, we can train and test by iterating a loop over dataSet1.CrossValidatedTrain, etc
#to get the 10 folds
#print(dataSet1.crossValidatedTrain[0])
#print(dataSet1.crossValidatedTrainOut[0])
GA = GeneticAlg(10, 2, 4, .1, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
GA.train(10000)
