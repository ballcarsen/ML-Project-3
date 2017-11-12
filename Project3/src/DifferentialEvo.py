#!/usr/bin/env python3
from .EvoAlg import EvoAlg
import random
import math
from .Tester import Tester as test

'''
Better for larger populations
Generational Replacement is typical
Mutate, then crossover
Process: 
    Mutation: <u> = <x1> + F(<x2> - <x3>), 1, 2, 3 randomly selected and all different
    Crossover: Cross <xi> and <u>
'''
class DifferentialEvolution(EvoAlg):

    def train(self, maxIterations, mutationFactor = 1, crossoverRate = 0.5):
        self.mf = mutationFactor
        self.cr = crossoverRate
        generations = 0
        while generations < maxIterations:
            print("Generation: %s" % generations)
            best, bestFit = self.getBest()
            print("Current Max Fitness: %s" % bestFit)
            children = []
            for i in range(len(self.population) - 1): # i for individual
                x = self.population[i]
                a, b, c = self.getIndividualsIndex(i)
                i1 = self.population[a]
                i2 = self.population[b]
                i3 = self.population[c]
                u = self.deMutate(i1, i2, i3)
                u = self.crossover(u, x)
                children.append(u if self.evalFitness(u) > self.evalFitness(x) else x)
            children.append(best)
            self.population = children
            self.printFitnessInfo()
            generations += 1
        return self.getBest()


    # See notes about DE mutation
    def deMutate(self, a, b, c):
        u = list(a)
        for i in range(len(a)):
            for j in range(len(a[i])):
                for k in range(len(a[i][j].weights)):
                    u[i][j].weights[k] = a[i][j].weights[k] + self.mf * (b[i][j].weights[k] - c[i][j].weights[k])
        return u

    def crossover(self, u, x):
        c = list(u) # c for child
        for i in range(len(u)):
            for j in range(len(u[i])):
                for k in range(len(u[i][j].weights)):
                    crossover = random.random()
                    if crossover <= self.cr: # trait from u
                        c[i][j].weights[k] = u[i][j].weights[k]
                    else: # trait from x
                        c[i][j].weights[k] = x[i][j].weights[k]
        return c
                        
    def printFitnessInfo(self):
        fitness = []
        for p in self.population:
            fitness.append(self.evalFitness(p))
        t = test(fitness)
        print(fitness)
        print("\nNew Max Fitness: %s" % t.get_max())
        print("Mean: %s\nStandard Deviation: %s" % (t.get_mean(), t.get_stdev()))

    def getBest(self):
        bestFit = self.evalFitness(self.population[0])
        bestInd = 0
        for i in range(1, len(self.population)):
            challenge = self.evalFitness(self.population[i])
            if challenge > bestFit:
                bestFit = challenge
                bestInd = i
        return self.population[bestInd], bestFit


    # Randomly select three individuals (index) from population
    def getIndividualsIndex(self, current):
        a = b = c = current
        while a == current:
            a = random.randint(0, len(self.population) - 1)
        while b == current or b == a:
            b = random.randint(0, len(self.population) - 1)
        while c == current or c == b or c == a:
            c = random.randint(0, len(self.population) - 1)
        return a, b, c
        

