#Child class of EvoAlg, the Genetic Algorithm
import math

from src.EvoAlg import EvoAlg
from src.Tester import Tester
import random

class GeneticAlg(EvoAlg): 
    
    # train our population over maxIterations using the given data
    def train(self, maxIterations):
        # for each generation
        genCount = 0
        sigma = self.calcSigma()
        while(genCount <= maxIterations):
            self.children = [] # reset children array
            genCount += 1 # increment generation count
            parents = self.select() # select parents using tournament selection
            # populate children via crossover
            for i in range(len(parents)): # for every parent
                if (i % 2 == 0):
                    if (i + 2 > len(parents)):
                        self.children.append(parents[i])
                    else:
                        childArr = self.crossover(parents[i],parents[i+1])
                        for child in childArr:
                            self.children.append(child)
            # mutate children

            for child in self.children:
                child = self.gaussMuatate(child, sigma)
            # replace members of self.population with self.children if the children are more fit
            self.replaceAll()
            print(self.test(self.inputData, self.expectedOut, self.getBestIndiv()) / float(len(self.inputData)))
    
    def gaussMuatate(self, child, sigma):
        for i in range(len(child) - 1):
            for k in range(len(child[i])):
                for j in range(len(child[i][k].weights)):
                    child[i][k].weights[j] = child[i][k].weights[j] + sigma

    def calcSigma(self):
        weights = []
        count = 0
        for c in range(len(self.population)):
            net = self.population[c]
            for i in range(len(net)):
                for k in range(len(net[i])):
                    for j in range(len(net[i][k].weights)):
                        weights.append(net[i][k].weights[j])
                        count+= 1
        T1 = Tester(weights)
        return math.sqrt(T1.get_variance())

    def select(self):
        parents = []
        for i in range(self.popSize):
            parents.append(self.tournamentSelection(self.popSize/20))
        return parents
        
    
    def tournamentSelection(self, k):
        competitors = [];
        # randomly select k competitors
        while(k > 0):
            index = random.randint(0,self.popSize-1)
            competitors.append(self.population[index])
            k -= 1
        #print(competitors)
        # compete the competitors to get winner and return winner
        winner = self.hostTournament(competitors)
        return winner
        
    # find best competitor
    def hostTournament(self, competitors):
        best = competitors[0]
        bestFitness = self.evalFitness(best)
        for competitor in competitors:
            if (self.evalFitness(competitor) > bestFitness):
                best = competitor
                bestFitness = self.evalFitness(best)
        return best
                
           
