from src.shared.forwardProp import ForwardProp
from pandas.tests.io.msgpack.test_read_size import test_correct_type_nested_array

def test(input, expectedOut, network):
    correct = 0 #incorrect by default
    fp = ForwardProp(network,input,expectedOut)
    # if classification is correct, return 1
    if (fp.getHypothesis() == expectedOut):
        correct = 1
    return correct