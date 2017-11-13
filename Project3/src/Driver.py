#energy 7
#indians 2
#HTRU2 2
#seeds 3
#shuttle 5

from src.shared.forwardProp import ForwardProp
from src.Data import Data
from src.backprop.backProp import BackProp
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array
from src.GeneticAlg import GeneticAlg
from src.EvoStrategy import EvoStrat
from src.backprop.bpAlg import BPAlg
from src.Tester import Tester

class Driver:

    def __init__(self):    
        self.numberOfClasses = 3 # will need to change this so it's not hard coded
        self.data = None
        self.expectedOut = None
        self.gABest = []
        self.evoAlgBest = []
        self.evoStraBest = []

    #Accepts and array of percentages correct for two different training methods


    #Accepts a file name, will return the data, and expeceted output in an array [data,out]
#Sample data set
#Pass number of classes
dataSet1 = Data(3)
#Pass file name
dataSet1.readData('seeds.txt')
#do the foldin
dataSet1.getFolds()
print('Data Set')
print(dataSet1.crossValidatedTrain[0])
print(dataSet1.crossValidatedTrainOut[0])
#dataSet2 = Data(7)
#dataSet2.readData('energy.txt')
#one set of data points, the first fold, we can train and test by iterating a loop over dataSet1.CrossValidatedTrain, etc
#to get the 10 folds
#print(dataSet1.crossValidatedTrain[0])
#print(dataSet1.crossValidatedTrainOut[0])

print("Begin GA")
GA = GeneticAlg(20, 2, 20, .7, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
GA.train(5)
print("Begin EVO")
EVO = EvoStrat(20, 2, 20, .7, dataSet1.crossValidatedTrain[0], dataSet1.crossValidatedTrainOut[0])
EVO.train(5)
Gbest = GA.getBestIndiv()
Epercent = []
Gpercent = []
Ebest = EVO.getBestIndiv()
Ecount = 0
Gcount = 0

for k in range(len(dataSet1.crossValidatedTest)):
    for i in range(len(dataSet1.crossValidatedTest[k])):
        Gcount += GA.test(dataSet1.crossValidatedTest[k][i], dataSet1.crossValidatedTestOut[k][i], Gbest)
    Gpercent.append(Gcount / len(dataSet1.crossValidatedTest[k]))
    Gcount = 0
print(Gpercent)

for k in range(len(dataSet1.crossValidatedTest)):
    for i in range(len(dataSet1.crossValidatedTest[k])):
        Ecount += EVO.test(dataSet1.crossValidatedTest[k][i],dataSet1.crossValidatedTestOut[k][i], Ebest)
    Epercent.append(Ecount / len(dataSet1.crossValidatedTest[k]))
    Ecount = 0
print(Epercent)



