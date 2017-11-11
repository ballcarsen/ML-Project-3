'''
Created on Nov 8, 2017

@author: Brett
'''
import unittest
from src.shared.forwardProp import ForwardProp
from src.backprop.bpNetCreator import BPNetCreator
import random
from src.shared.printNetwork import NetworkPrinter


class Test(unittest.TestCase):
    
    def setUp(self):
        inNum = 2
        outNum = 1
        netCreator = BPNetCreator(1,2,inNum,outNum)
        network = netCreator.create()
        inputs = [-.9, .2]
        self.expectedOutputs = [1,0]
        self.fp = ForwardProp(network,inputs,self.expectedOutputs)
        netPrinter = NetworkPrinter()
        netPrinter.printNet(network)
        self.hypothesis = self.fp.getHypothesis()
        pass


    def testHypothesisBetweenZeroAndLengthOfOutputs(self):     
            self.assertTrue(self.hypothesis < len(self.expectedOutputs) and self.hypothesis >= 0, "hypothesis is not within range")
            
    def testHypothesisIsWholeNumber(self):
        self.assertEqual(self.hypothesis % 1, 0, "hypothesis is not a whole number")
        
