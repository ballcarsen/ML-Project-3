#energy 7
#indians 2
#HTRU2 2
#seeds 3
#shuttle 5

from src.shared.forwardProp import ForwardProp
from src.Data import Data
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.GeneticAlg import GeneticAlg
from src.EvoStrategy import EvoStrat
from src.Tester import Tester

class Driver:

    def __init__(self):    
        self.numberOfClasses = 3 # will need to change this so it's not hard coded
        self.data
        self.expectedOut
        self.gABest = []
        self.evoAlgBest = []
        self.evoStraBest = []

    #Accepts and array of percentages correct for two different training methods


    #Accepts a file name, will return the data, and expeceted output in an array [data,out]
#Sample data set
#Pass number of classes
dataSet1 = Data(2)
#Pass file name
dataSet1.readData('HTRU2.txt')
#do the foldin
dataSet1.getFolds()
dataSet2 = Data(7)
dataSet2.readData('energy.txt')
#one set of data points, the first fold, we can train and test by iterating a loop over dataSet1.CrossValidatedTrain, etc
#to get the 10 folds
#print(dataSet1.crossValidatedTrain[0])
#print(dataSet1.crossValidatedTrainOut[0])
out1 = open('outHTRU2.txt', 'w')
a = "GA"
a += '\t'
a += 'EVO\n'
for k in range(3):
    for i in range(len(dataSet1.crossValidatedTrainOut)):
        GA = GeneticAlg(50, 2, 20, .3, dataSet1.crossValidatedTrain[i], dataSet1.crossValidatedTrainOut[i])
        GA.train(100)
        EVO = EvoStrat(50, 2, 20, .3, dataSet1.crossValidatedTrain[i], dataSet1.crossValidatedTrainOut[i])
        EVO.train(100)
        s = str(GA.evalFitness(GA.getBestIndiv()))
        s += '\t'
        s += str(EVO.evalFitness(GA.getBestIndiv()))
        s += '\n'
        out1.write(s)
out1.close

out2 = open('outENERGY.txt', 'w')
a = "GA"
a += '\t'
a += 'EVO\n'
for k in range(3):
    for i in range(len(dataSet1.crossValidatedTrainOut)):
        GA = GeneticAlg(50, 2, 20, .3, dataSet2.crossValidatedTrain[i], dataSet2.crossValidatedTrainOut[i])
        GA.train(100)
        EVO = EvoStrat(50, 2, 20, .3, dataSet2.crossValidatedTrain[i], dataSet2.crossValidatedTrainOut[i])
        EVO.train(100)
        s = str(GA.evalFitness(GA.getBestIndiv()))
        s += '\t'
        s += str(EVO.evalFitness(GA.getBestIndiv()))
        s += '\n'
        out2.write(s)
out2.close()

GA = GeneticAlg(100, 2, 20, .3, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
GA.train(50)
EVO = EvoStrat(100, 2, 50, .1, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
EVO.train(50)
print('Percent Correct', GA.evalFitness(GA.getBestIndiv()))

