from src.shared.forwardProp import ForwardProp
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array

class Driver:

    def __init__(self):    
        self.numberOfClasses = 3 # will need to change this so it's not hard coded

    def test(self, input, expectedOut, network):
        correct = 0 #incorrect by default
        fp = ForwardProp(network,input,self.toArrayRep(expectedOut))
        # if classification is correct, return 1
        if (fp.getHypothesis() == expectedOut):
            correct = 1
        return correct

    
    def toArrayRep(self, index):
        array = []
        for i in range(self.numberOfClasses):
            if (i == index):
                array.append(1)
            else:
                array.append(0)
        return array
