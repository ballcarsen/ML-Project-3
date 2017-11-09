#Child class of EvoAlg, the Genetic Algorithm
from src import EvoAlg
import random

class GeneticAlg(EvoAlg): 
    
    # train our population over maxIterations using the given data
    def train(self, maxIterations, inputData, outputData):
        self.inputData = inputData
        self.outputData = outputData
        # for each generation
        while(self.genCount <= maxIterations):
            self.currentGen += 1 # increment generation count
            parents = self.select() # select parents using tournament selection
            children = []
            # populate children via crossover
            for i in range(len(parents)): # for every parent
                if (i % 2 == 0):
                    if (i + 2 > len(parents)):
                        children.append(parents[i])
                    else:
                        childArr = self.crossover(parents[i],parents[i+1])
                        for child in childArr:
                            children.append(child)
            children = [self.gaussMuatate(child) | child in children]
            self.population = children # generational replacement
    
    def gaussMuatate(self, child):
        return child
        
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
        print(competitors)
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
                
           
