from src.GeneticAlg import GeneticAlg
from src.Tester import Tester
import random
import math
#Evolutionary Strategy
class EvoStrat(GeneticAlg):
    #Chagnges sigma
    def updateVar(self, length, sigma):
        u = random.uniform(0,sigma)
        if length == 0:
            length = 1
        s = sigma * math.exp(u/math.sqrt(length))
        return s
    #muttation
    def gaussMuatate(self, child):
        for i in range(len(child) - 1):
            for k in range(len(child[i])):
                for j in range(len(child[i][k].weights) - 1):
                    l = child[i][j].weights
                    sigma = self.var(l)
                    child[i][k].weights[j] = child[i][k].weights[j] + self.updateVar(len(l),sigma)
    #get variance of the weights
    def var(self, l):
        T1 = Tester(l)
        return  math.sqrt(T1.get_variance())
    #train the networks
    def train(self, maxIterations):
        # for each generation
        genCount = 0
        while(genCount <= maxIterations):
            self.children = [] # reset children array
            genCount += 1 # increment generation count
            parents = self.select() # select parents using tournament selection
            # populate children via crossover
            for i in range(len(parents)): # for every parent
                # by steps of two
                if (i % 2 == 0):
                    # if we have reached the end of the array, just select last element
                    if (i + 2 > len(parents)):
                        self.children.append(parents[i])
                    # otherwise cross parents and add children to children array
                    else:
                        childArr = self.crossover(parents[i],parents[i+1])
                        for child in childArr:
                            self.children.append(child)
            # mutate children
            for child in self.children:
                child = self.gaussMuatate(child)
            # replace members of self.population with self.children if the children are more fit
            self.replaceAll()
            print(self.evalFitness(self.getBestIndiv()), "performance")