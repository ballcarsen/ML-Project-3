from src.GeneticAlg import GeneticAlg
import math

class EvoStrat(GeneticAlg):

    def updateVar(self, length, sigma):
        u = math.random(0,sigma)
        s = sigma * math.exp(u/math.sqrt(length))
        return s
    def gaussMuatate(self, child, sigma):
        for i in range(len(child) - 1):
            for k in range(len(child[i])):
                for j in range(len(child[i][k].weights)):
                    l = len(child[i][j].weights)
                    child[i][k].weights[j] = child[i][k].weights[j] + self.updateVar(l,sigma)
