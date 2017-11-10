#Child class of EvoAlg, the Genetic Algorithm
from src.EvoAlg import EvoAlg
import random

class GeneticAlg(EvoAlg): 
    
    # train our population over maxIterations using the given data
    def train(self, maxIterations):
        # for each generation
        genCount = 0
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
                child = self.gaussMuatate(child)
            # replace members of self.population with self.children if the children are more fit
            self.replaceAll()
    
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
                
           
