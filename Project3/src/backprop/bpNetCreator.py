from src.shared import node

class BPNetCreator:
    def __init__(self, hiddenLayerNum, nodesInHLNum, inNum, outNum):
        self.hiddenLayerNum = hiddenLayerNum
        self.nodesInHLNum = nodesInHLNum
        self.inNum = inNum
        self.outNum = outNum
        self.network = []
       
    # create and return network of nodes
    def create(self):
        # input layer:
        self.network.append([])
        # a node for each input 
        for i in range(self.inNum):
            # no input weights for first layer
            self.network[0].append(node.BPNode(0))
        # append bias node
        self.network[0].append(node.BiasNode(0))

        # hidden layers:
        # j represents layer
        for j in range(1,self.hiddenLayerNum + 1):
            # create new column
            self.network.append([])
            # for every node in the layer
            for i in range(self.nodesInHLNum): 
                # if first hidden layer
                if (j == 1):
                    # use number of inputs + 1 for weight array length
                    self.network[j].append(node.BPNode(self.inNum + 1))
                # other hidden layers
                else:
                    self.network[j].append(node.BPNode(self.nodesInHLNum + 1))
            # append bias node at end
            self.network[j].append(node.BiasNode(0))
        # output layer node: uses no activation function, just sum
        self.network.append([])
        for i in range(self.outNum):
            if (self.hiddenLayerNum != 0):
                self.network[self.hiddenLayerNum + 1].append(node.Node(self.nodesInHLNum + 1))
            else:
                self.network[self.hiddenLayerNum + 1].append(node.Node(self.inNum + 1))
        return self.network
    