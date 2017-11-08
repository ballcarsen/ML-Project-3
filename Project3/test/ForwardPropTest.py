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
        outNum = 2
        netCreator = BPNetCreator(2,5,inNum,outNum)
        network = netCreator.create()
        inputs = []
        for i in range(inNum):
            inputs.append(random.uniform(0,1))
        print("inputs", inputs)
        self.expectedOutputs = []
        for i in range(outNum):
            self.expectedOutputs.append(random.randint(0,1))  
        print("expected outputs", self.expectedOutputs)
        self.fp = ForwardProp(network,inputs,self.expectedOutputs)
        self.hypothesis = self.fp.getHypothesis()
        pass


    def testHypothesisOneOrZero(self):     
        for prediction in self.hypothesis:
            self.assertTrue(prediction == 1 or prediction == 0, "hypothesis didn't return a whole number")
            
    def testHypothesisSize(self):
        self.assertEqual(len(self.expectedOutputs), len(self.hypothesis), "length of hypothesis not equal to expected outputs")
        
    
  
