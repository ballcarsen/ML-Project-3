'''
Created on Nov 8, 2017

@author: Brett
'''
import unittest
from src.shared.forwardProp import ForwardProp
from src.backprop.bpNetCreator import BPNetCreator
import random


class Test(unittest.TestCase):
    
    def setUp(self):
        inNum = 3
        outNum = 4
        netCreator = BPNetCreator(2,5,inNum,outNum)
        network = netCreator.create()
        inputs = []
        for i in range(inNum):
            inputs.append(random.uniform(0,1))
        print("inputs", inputs)
        self.expectedOutputs = [1,0,0,0]
        print("expected outputs", self.expectedOutputs)
        self.fp = ForwardProp(network,inputs,self.expectedOutputs)
        self.hypothesis = self.fp.getHypothesis()
        pass


    def testHypothesisBetweenZeroAndLengthOfOutputs(self):     
            self.assertTrue(self.hypothesis < len(self.expectedOutputs) and self.hypothesis >= 0, "hypothesis is not within range")
            
    def testHypothesisIsWholeNumber(self):
        self.assertEqual(self.hypothesis % 1, 0, "hypothesis is not a whole number")
        
