import math
from src.shared.printNetwork import NetworkPrinter

class ForwardProp:
    def __init__(self, network, inputs, expectedOuts):
        self.expectedOuts = expectedOuts
        self.network = network
        # netPrinter = NetworkPrinter()
        # print(" -------------- INSIDE FORWARDPROP ---------------------")
        # netPrinter.printNet(network)
        self.inputs = inputs
        self.hypothesis = [] # will store list of outputs
        # calculate hypothesis
        self.calcHypothesis()
        
    def calcHypothesis(self):
        prevActivs = [] # list of activations from previous layer
        currentActivs = [] # to be used as previous activations in next iteration
        # input layer
        for i in range(len(self.network[0])-1):
            # initial activations are based on inputs
            self.network[0][i].setActiv(self.inputs[i])
            prevActivs.append(self.network[0][i].getActiv())
        # bias node: set activation to 1
        self.network[0][len(self.network[0]) - 1].setActiv(1)
        prevActivs.append(self.network[0][len(self.network[0]) - 1].getActiv())
        # hidden and output layers
        for j in range(1, len(self.network)):
            for i in range(len(self.network[j])):
                # set activations based on previous activations
                self.network[j][i].calcActiv(prevActivs)
                # store activations in currentActivs list
                currentActivs.append(self.network[j][i].getActiv())
                # if we are in output layer, set output delta
                if (j == len(self.network)-1):
                    #self.network[j][i].setErrorDiff(self.network[j][i].getActiv()-self.expectedOuts[i])
                    self.network[j][i].setDelta(self.calcClassificationOutputDelta(self.network[j][i].getActiv(),self.expectedOuts[i]))
            # prevActivs takes on values in currentActivs for next layer
            prevActivs = currentActivs
            currentActivs = []
        # outputs are final activations
        self.hypothesis = prevActivs
        
    # output delta for classification
    def calcClassificationOutputDelta(self,output,expected):
        return (output - expected) * output * (1 - output)
    
    # output delta for function approximation
    def calcFunctApproxOutputDelta(self,output,expected):
        return output - expected
    
    # for use in test phase
    def getTotalSquaredError(self):
        error = 0
        for i in range(len(self.expectedOuts)):
            error += math.pow((self.expectedOuts[i] - self.hypothesis[i]), 2)
        return error
    
    # for debugging (delta's already set for output layer)
    def getErrorArray (self):
        errors = []
        for i in range(len(self.expectedOuts)):
            # output deltas are simply difference between the hypothesis and the expected output
            errors.append(self.hypothesis[i] - self.expectedOuts[i]) 
        return errors
    
    # return whatever class is most likely, based on the node with the highest activation
    def getHypothesis(self):
        # convert hypothesis to class 1 or 0
        bestHypothesis = -1
        bestIndex = -1
        for i in range(len(self.hypothesis)):
            if (self.hypothesis[i] > bestHypothesis): 
                bestHypothesis = self.hypothesis[i]
                bestIndex = i
        return bestIndex
            
            
