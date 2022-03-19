import numpy as np

from typing import List, Callable



class NeuralNetwork:
    '''
    This class includes all functionalities for a neural network.
    '''

    def __init__(self, dimensions: List[int], weights: List['numpy_array'] = [], biases: List['numpy_array'] = [], activation_functions: List[Callable] = []) -> None:

        # list of dimensions of the neural network
        self.dimensions = dimensions

        # number of layers the neural network has
        self.layers: int = len(self.dimensions)

        # list of all the weight matrices for the neural net
        # entry at i is w^{i + 1}
        self.weights = weights

        # list of all the bias vectors for the neural net
        # entry at i is b^{i + 1}
        self.biases = biases

        # list containing all the activation functions for each layer
        # entry at i is \sigma^{i + 1}
        self.activation_functions = activation_functions


    def initialize_network(self) -> None:
        '''
        This method randomly initializes the weights and the biases.
        '''

        # initialize the weights matrices w^{l} with dimensions dim(l) x dim(l - 1)
        for l, dim in enumerate(self.dimensions[1:], 1):
            self.weights.append(np.random.rand(dim, self.dimensions[l - 1]))

        # initialite the bias vectors b^{l} with dimensions dim(l)
        for l in self.dimensions[1:]:
            self.biases.append(np.random.rand(l))


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
