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
from src.backprop.bpAlg import BPAlg
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
dataSet1 = Data(7)
#Pass file name
dataSet1.readData('energy.txt')
#do the foldin
dataSet1.getFolds()
#dataSet2 = Data(7)
#dataSet2.readData('energy.txt')
#one set of data points, the first fold, we can train and test by iterating a loop over dataSet1.CrossValidatedTrain, etc
#to get the 10 folds
#print(dataSet1.crossValidatedTrain[0])
#print(dataSet1.crossValidatedTrainOut[0])

out1 = open('outEnergy.txt', 'w')
a = "GA"
a += '\t'
a += 'EVO\n'

GA = GeneticAlg(20, 2, 20, .5, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
GA.train(10)
EVO = EvoStrat(20, 2, 20, .5, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
EVO.train(10)
Gbest = GA.getBestIndiv()
Epercent = []
Gpercent = []
Ebest = EVO.getBestIndiv()
Ecount = 0
Gcount = 0

for k in range(len(dataSet1.crossValidatedTest)):
    for i in range(len(dataSet1.crossValidatedTest[k])):
        Gcount += GA.test(dataSet1.crossValidatedTest[k][i], dataSet1.crossValidatedTestOut[k][i], Gbest)
        Ecount += EVO.test(dataSet1.crossValidatedTest[k][i],dataSet1.crossValidatedTestOut[k][i], Ebest)
    Gpercent.append(Gcount / len(dataSet1.crossValidatedTest[k]))
    Epercent.append(Ecount / len(dataSet1.crossValidatedTest[k]))
    Gcount = 0
    Ecount = 0
print(Gpercent)
print(Epercent)




'''
out2 = open('outENERGY.txt', 'w')
a = "GA"
a += '\t'
a += 'EVO\t'
a+= 'BP\n'
for i in range(len(dataSet1.crossValidatedTrainOut)):
    GA = GeneticAlg(50, 2, 20, .3, dataSet2.crossValidatedTrain[i], dataSet2.crossValidatedTrainOut[i])
    GA.train(10)
    EVO = EvoStrat(50, 2, 20, .3, dataSet2.crossValidatedTrain[i], dataSet2.crossValidatedTrainOut[i])
    EVO.train(10)
    backprop = BPAlg()
    trainedNetwork = backprop.train(dataSet2.crossValidatedTrain[i], dataSet2.crossValidatedTrainOut[i], 2, 20)
    backprop = BPAlg()
    trainedNetwork = backprop.train(dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0], 2, 20)
    s = str(GA.test(dataSet2.crossValidatedTest[i],dataSet2.crossValidatedTestOut[i], GA.getBestIndiv()))
    s += '\t'
    s += str(EVO.evalFitness(GA.getBestIndiv()))
    s += '\n'
    out2.write(s)


GA = GeneticAlg(50, 2, 20, .3, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
GA.train(10)
EVO = EvoStrat(50, 2, 20, .3, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
EVO.train(10)
print('Percent Correct', GA.evalFitness(GA.getBestIndiv()))

# train backprop
backprop = BPAlg()
trainedNetwork = backprop.train(dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0], 2, 20)
#test backprop
print(backprop.test(dataSet1.crossValidatedTest[0], dataSet1.crossValidatedTestOut[0],trainedNetwork))
'''