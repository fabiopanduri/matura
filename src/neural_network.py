import numpy as np

from typing import List, Callable


def sigmoid(x):
    '''
    Sigmoid activation function.
    '''

    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    '''
    Derivative of the sigmoid activation function.  
    '''

    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    '''
    ReLU activation function.
    '''

    return max(0, x)

def ReLU_derivative(x):
    '''
    Derivative of the ReLU activation function.
    '''

    return 1 if x > 0 else 0

def cost_function_derivative(activation, y):
    '''
    Derivative of the cost function with respecto to the activation.
    Here derivative of Mean squared error.
    '''

    return activation - y


class NeuralNetwork:
    '''
    This class includes all functionalities for a neural network.
    '''

    def __init__(self, dimensions: List[int], eta: float, weights: List['numpy_array'] = [], biases: List['numpy_array'] = [], activation_functions: List[Callable] = []) -> None:

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
        
        # learning rate eta for gradient descent
        self.eta = eta


    def initialize_network(self) -> None:
        '''
        This method randomly initializes the weights and the biases.
        '''

        # add empty array to self.weights so that the index corresponds to the layer
        self.weights = [np.array([])]

        # initialize the weights matrices w^{l} with dimensions dim(l) x dim(l - 1)
        for l, dim in enumerate(self.dimensions[1:], 1):
            self.weights.append(np.random.uniform(-1, 1, (dim, self.dimensions[l - 1])))

        # add empty array to self.weights so that the index corresponds to the layer
        self.biases = [np.array([])]

        # initialite the bias vectors b^{l} with dimensions dim(l)
        for dim_l in self.dimensions[1:]:
            self.biases.append(np.random.uniform(-1, 1, dim_l))


    def feed_forward(self, input_vector: 'numpy_array') -> 'numpy_array':
        '''
        This method feeds the input_vector through the network.
        '''

        activation = input_vector
        print(f'{activation=}')
        # for each layer compute the activation
        for l in range(1, self.layers):
            z_l = np.matmul(self.weights[l], activation) + self.biases[l]
            activation = self.activation_functions[l](z_l)
            print(f'{l=}')
            print(f'{activation=}')
            print(f'{self.weights=}')
            print('')

        # return the activation for the output layer
        return activation


    def stochastic_gradient_descent(self, training_set) -> None:
        '''
        This method implements the stochastic gradient descent algorithm.
        '''

        # initialize lists to store the sum of errors for each weight and bias
        weight_delta_sum = [np.zeros(self.dimensions[l]) for l in range(self.layers)]
        bias_delta_sum = [np.zeros(self.dimensions[l]) for l in range(self.layers)]

        # iterate over all training examples
        for training_example in training_set:
            activation = training_example[0]

            # set up list to store all the z vectors for later use
            # initialized with z vector at 0 = 0 for indexing purposes
            z_list = [0]

            # set up a list to store all the activation vectors for each later
            activation_list = []
            activation_list.append(activation)

            # calculate the activation and z vector for all the layers
            for l in range(1, self.layers):
                # compute the z vector and store it in the z_list 
                z = np.dot(self.weights[l], activation) + self.biases[l]
                z_list.append(z)

                # compute the activation of the l-th layer and add it to the activation list
                activation = self.activation_functions[l](activation)
                activation_list.append(activation)

            # calculate the error of the last layer
            delta_L = cost_function_derivative(activation, training_example[1]) * sigmoid_derivative(z)

            # calculate the error of each other layer
            delta = [np.zeros(self.dimensions[l]) for l in range(self.layers)]
            delta[-1] = delta_L.reshape(1, 4)
            for l in range(self.layers - 2, 0, -1):
                #print(delta[-1])
                #print(z[l]
                #print(l, np.matmul(self.weights[l + 1].T, delta[l + 1]), self.weights[l + 1].T, delta[l + 1])
                delta[l] = np.dot(self.weights[l + 1].T, delta[l + 1]) * sigmoid_derivative(z_list[l])

                # update the sum of errors for weights and biases for each layer
                weight_delta_sum[l] = weight_delta_sum[l] + np.dot(activation_list[l - 1].T, delta[l])
                bias_delta_sum[l]  = bias_delta_sum[l] + delta[l]


        # update the weights and biases according to the calculated errors
        for l in range(1, self.layers):
            self.weights[l] = self.weights[l] - weight_delta_sum[l] * self.eta / len(training_set)
            self.biases[l] = self.biases[l] - bias_delta_sum[l] * self.eta / len(training_set)





def main():
    NN = NeuralNetwork([2, 2, 1], 0.9, weights = [np.array([]), np.array([[0, 1], [1, 0]]), np.array([2, 3])],
        biases = [np.array([]), np.array([0]), np.array([0])],
        activation_functions = [0, np.vectorize(lambda x: x), np.vectorize(lambda x: x)])
    #NN.initialize_network()
    print(NN.feed_forward(np.array([2, 1])))

if __name__ == '__main__': main()
