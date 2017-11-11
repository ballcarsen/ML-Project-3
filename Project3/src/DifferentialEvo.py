#!/usr/bin/env python3
from src import EvoAlg
import random
import math

'''
Better for larger populations
Generational Replacement is typical
Mutate, then crossover
Process: 
    Mutation: <u> = <x1> + F(<x2> - <x3>), 1, 2, 3 randomly selected and all different
    Crossover: Cross <xi> and <u>
'''
class DifferentialEvolution(EvoAlg):

    def __init__(self, mutationFactor, crossoverRate):
        self.mf = mutationFactor
        self.cr = crossoverRate

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
                        


    def train(self, maxIterations):
        generations = 0
        while generations < maxIterations:
            for i in range(len(self.population)): # i for individual
                a, b, c = self.getIndividualsIndex()
                i1 = self.population(a)
                i2 = self.population(b)
                i3 = self.population(c)
                u = deMutate(i1, i2, i3)
                u = crossover(u, x)


    # Randomly select three individuals (index) from population
    def getIndividualsIndex(self, current):
        a = b = c = current
        while a == current:
            a = random.random() * (len(self.population) - 1)
        while b == current or b == a:
            b = random.random() * (len(self.population) - 1)
        while c == current or c == b or c == a:
            c = random.random() * (len(self.population) - 1)
        return a, b, c
        

