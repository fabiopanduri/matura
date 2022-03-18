import numpy as np

from typing import List


class NeuralNetwork:
    def __init__(self, dimensions: List[int], weights: 'numpy_array', bias: 'numpy_array') -> None:
        self.dimensions = dimensions
        self.weights = weights
        self.bias = bias

    def initialize_network(self) -> None:
        '''
        This method randomly initializes the weights and the biases.
        '''
         
        pass

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
    pass

if __name__ == '__main__': main()
