#Super class for the three evolutionary training algorithms
from src.backprop.bpNetCreator import BPNetCreator
from src.shared.forwardProp import ForwardProp
import random

import random
from src.shared.printNetwork import NetworkPrinter
from src import helper
class EvoAlg:
    def __init__(self, popSize, hiddenLayerNum, nodesInHLNum, crossoverRate, inputData, expectedOut):
        self.inputData = inputData
        self.expectedOut = expectedOut
        self.population = None
        self.popSize = popSize
        #the threshold for the crossover probability. If .5 attributes will have a 50% chance of cross over
        self.crossoverRate = crossoverRate
        self.hiddenLayerNum = hiddenLayerNum
        self.nodesInHLNum = nodesInHLNum
        self.children = []
        self.netCreator = BPNetCreator(self.hiddenLayerNum, self.nodesInHLNum, len(self.inputData[0]), len(self.expectedOut[0]))
        # initialize population
        self.initPop()


    #Creates popSize networks with random weights
    def initPop(self):
        self.population = []
        #Initializes a new network creator from the backprop code
        for i in range(self.popSize):
            #Creates popSize MLP's and adds them to the population list
            self.population.append(self.netCreator.create())
            

    #Calculates the percent correctly classified by a network
    def evalFitness(self, network):
        totalCorrect = 0
        #Assuming driver has a test, that returns 1 if classified correctly, 0 if not
        for i in range(len(self.inputData)):
            totalCorrect += self.test(self.inputData[i], self.expectedOut[i], network)
        return(totalCorrect / len(self.inputData))


    def test(self, input, expectedOut, network):
        correct = 0 #incorrect by default
        fp = ForwardProp(network, input, expectedOut)
        if 1 not in expectedOut:
            return 0
        if (fp.getHypothesis() == expectedOut.index(1)):
            correct = 1
        return correct


    #will perform binary crossover on two networks, thought this could also just calculate the mask
    #this will return two children, I think mutation comes after crossover?
    def crossover(self, parent1, parent2):
        #creates two networks that will be used as children
        child1 = self.netCreator.create()
        child2 = self.netCreator.create()
        #Itterates through the layers, ignoring the input and output layer
        for i in range(1, (len(parent1) - 1)):
            #itterates through the nodes in each layer
            for k in range(self.nodesInHLNum):
                #for each weight in the node
                for j in range(len(parent1[i][k].weights)):
                    #random uniform cross over for each attribute of the individual(weight)
                    weight1 = parent1[i][k].weights[j]
                    weight2 = parent2[i][k].weights[j]
                    #children get a certain parents weight
                    if random.uniform(0,1) < self.crossoverRate:
                        child1[i][k].weights[j] = weight2
                        child2[i][k].weights[j] = weight1
                    #children get the other parents weight
                    else:
                        child1[i][k].weights[j] = weight1
                        child2[i][k].weights[j] = weight2
        #returns both the children, to be mutated and used in replacement
        return [child1, child2]

   # replaces any parents with their children if their children have better fitness
    def replaceAll(self):
        for i in range(self.popSize):
            #Fitness of the parent
            fitP = self.evalFitness(self.population[i])
            print("parent fitness: ", fitP)
            #Fitness of the child
            fitC = self.evalFitness(self.children[i])
            print("child fitness", fitC)
            # if the child's fitness is greater, replace the parent
            if fitC > fitP:
                self.population[i] = self.children[i]
        
    #itterates through the population, retuns the individual with the highest percent correct
    def getBestIndiv(self):
        maxFit = 0.0000
        best = self.population[0]
        for indiv in self.population:
            fit = self.evalFitness(indiv)
            if fit > maxFit:
                maxFit = fit
                best = indiv
        return best
    
    def train(self, maxiterations, data):
        pass
        
        
