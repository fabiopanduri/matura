import numpy as np

from typing import List, Callable


def sigmoid(x):
    '''
    sigmoid activation function 
    '''

    return 1 / (1 + np.exp(-x))

def ReLU(x):
    '''
    ReLU activation function
    '''

    return max(0, x)

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
        # entry at l is w^{l}
        self.weights = weights

        # list of all the bias vectors for the neural net
        # entry at l is b^{l}
        self.biases = biases

        # list containing all the activation functions for each layer
        # entry at l is \sigma^{l}
        self.activation_functions = activation_functions


    def initialize_network(self) -> None:
        '''
        This method randomly initializes the weights and the biases.
        '''

        # add empty array to self.weights so that the index corresponds to the layer
        self.weights = [np.array([])]

        # initialize the weights matrices w^{l} with dimensions dim(l) x dim(l - 1)
        for l, dim in enumerate(self.dimensions[1:], 1):
            self.weights.append(np.random.rand(dim, self.dimensions[l - 1]))

        # add empty array to self.weights so that the index corresponds to the layer
        self.biases = [np.array([])]

        # initialite the bias vectors b^{l} with dimensions dim(l)
        for dim_l in self.dimensions[1:]:
            self.biases.append(np.random.rand(dim_l))


    def feed_forward(self, input_vector: 'numpy_array') -> 'numpy_array':
        '''
        This method feeds the input_vector through the network.
        '''

        activation = input_vector
        # for each layer compute the activation
        for l in range(1, self.layers):
            z_l = np.dot(self.weights[l], activation) + self.biases[l]
            activation = self.activation_functions[l](z_l)

        # return the activation for the output layer
        return activation


    def backpropagation(self, cost: float) -> None:
        '''
        This method is used to update the weights and bias of the neral network.
        '''

        pass

    


def main():
    NN = NeuralNetwork([3, 5, 2])
    print(NN.initialize_network())

if __name__ == '__main__': main()
