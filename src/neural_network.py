import numpy as np

from typing import List



class NeuralNetwork:
    def __init__(self, dimensions: List[int], weights: List['numpy_array'] = [], biases: List['numpy_array'] = []) -> None:
        self.dimensions = dimensions
        self.weights = weights
        self.biases = biases

    def initialize_network(self) -> None:
        '''
        This method randomly initializes the weights and the biases.
        '''

        # initialize the weights matrices w^{l} with dimensions dim(l) x dim(l - 1)
        for i, dim in enumerate(self.dimensions[1:], 1):
            self.weights.append(np.random.rand(dim, self.dimensions[i - 1]))

        # initialite the bias vectors b^{l} with dimensions dim(l)
        for i in self.dimensions[1:]:
            self.biases.append(np.random.rand(i))


    def feed_forward(self, input_vector: 'numpy_array') -> 'numpy_array':
        '''
        This method feeds the input_vector through the network.
        '''



        pass

    def backpropagation(self, cost: float) -> None:
        '''
        This method is used to update the weights and bias of the neral network.
        '''

        pass

    


def main():
    NN = NeuralNetwork([3, 5, 2])
    print(NN.initialize_network())

if __name__ == '__main__': main()
