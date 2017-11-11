from src.shared.forwardProp import ForwardProp
from src.Data import Data
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.GeneticAlg import GeneticAlg
from src.Tester import Tester

    #Accepts a file name, will return the data, and expeceted output in an array [data,out]
#Sample data set
#Pass number of classes
dataSet1 = Data(6)
#Pass file name
dataSet1.readData('energyTestCropped.txt')
#do the foldin
dataSet1.getFolds()
#one set of data points, the first fold, we can train and test by iterating a loop over dataSet1.CrossValidatedTrain, etc
#to get the 10 folds
#print(dataSet1.crossValidatedTrain[0])
#print(dataSet1.crossValidatedTrainOut[0])
GA = GeneticAlg(1, 1, 6, .5, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
GA.train(0)

