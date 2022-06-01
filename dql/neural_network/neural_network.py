# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import datetime
import os

import numpy as np

import dql.neural_network.activation_functions as af


def mean_squared_error_derivative(activation, y):
    '''
    Derivative of the cost function (mean squared error) with respect to the activation.
    Here the derivative of Mean squared error.
    '''

    return activation - y


class NeuralNetwork:
    '''
    This class includes all functionalities for a neural network.
    '''

    def __init__(self,
                 dimensions,
                 eta,
                 activation_functions=[],
                 weights=[],
                 biases=[]
                 ) -> None:

        self.dimensions = dimensions

        self.layers = len(self.dimensions)

        # entry at l is w^{l}
        self.weights = weights

        # entry at l is b^{l}
        self.biases = biases

        self.activation_functions_names = activation_functions

        # entry at l is \sigma^{l}
        self.activation_functions = list(map(np.vectorize,
                                             [af.activation_functions[name]
                                                 for name in activation_functions]
                                             ))

        # entry at l is \sigma^{l}\prime
        self.activation_functions_derivatives = list(map(np.vectorize,
                                                         [af.activation_functions_derivatives[name]
                                                             for name in activation_functions]
                                                         ))

        self.eta = eta

    def initialize_network(self) -> None:
        '''
        This method randomly initializes the weights and the biases.
        '''

        # add empty array to self.weights so that the index l corresponds to the l-th layer
        self.weights = [np.array([])]

        for l, dim in enumerate(self.dimensions[1:], 1):
            self.weights.append(
                np.random.uniform(0, 1, (dim, self.dimensions[l - 1]))
            )

        # add empty array to self.weights so that the index l corresponds to the l-th layer
        self.biases = [np.array([])]

        for dim_l in self.dimensions[1:]:
            self.biases.append(np.random.uniform(0, 1, dim_l))

    def save_network(self, file_name: str = None) -> None:
        '''
        This method saves the current network to a file.
        It will be a .npz file containing all important data, namely:
         - dimensions
         - weights
         - biases
         - activation functions
         - learning rate eta
        '''

        if file_name == None:
            file_name = f'NN_saves/NN-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'

        if os.path.exists(file_name):
            inp = input(
                f'[WARNING] The file {file_name} already exists. Do you want to proceed? [y/n] ').lower()
            inp = 'n' # Remove this if input should be gathered. 
            while True:
                if inp == 'y':
                    print(f'[INFO] Saving to {file_name}...')
                    break
                elif inp == 'n':
                    print('[INFO] Saving aborted')
                    return
                else:
                    inp = input(
                        f'Invalid answer. Do you want to proceed? [y/n] ').lower()

        np.savez_compressed(file_name,
                            dimensions=self.dimensions,
                            weights=self.weights,
                            biases=self.biases,
                            activation_functions=self.activation_functions_names,
                            eta=self.eta
                            )

        print(f'[INFO] Saved data to \'{file_name}\'')

    @classmethod
    def load_network(cls, file_name: str) -> None:
        '''
        This method loads the current network from a file.
        '''

        if not os.path.exists(file_name):
            print('[ERROR] The specified file does not exist')

        data = np.load(file_name, allow_pickle=True)

        new_instance = cls(list(data['dimensions']),
                           float(data['eta']),
                           weights=data['weights'],
                           biases=data['biases'],
                           activation_functions=list(
                               data['activation_functions']),
                           )

        print(f'[INFO] loaded Neural Network from \'{file_name}\'')

        return new_instance

    def feed_forward(self, input_vector: 'numpy_array') -> 'numpy_array':
        '''
        This method feeds the input_vector through the network.
        '''

        activation = input_vector
        for l in range(1, self.layers):
            z_l = np.dot(self.weights[l], activation) + self.biases[l]
            activation = self.activation_functions[l](z_l)

        return activation

    def stochastic_gradient_descent(self, training_batch) -> None:
        '''
        This method implements the stochastic gradient descent algorithm.
Input: Training batch of form [(in, target_out), (in, target_out), ...]
        '''

        bias_delta_sum = [np.zeros(self.dimensions[l])
                          for l in range(self.layers)]

        weight_delta_sum = [
            0] + [np.zeros((self.dimensions[l], self.dimensions[l - 1])) for l in range(1, self.layers)]

        for training_example in training_batch:
            activation = training_example[0]

            # initialized with z vector at 0 equal to 0 for indexing purposes
            z_list = [0]

            activation_list = []
            activation_list.append(activation)

            for l in range(1, self.layers):
                z = np.dot(self.weights[l], activation) + self.biases[l]
                z_list.append(z)

                activation = self.activation_functions[l](z)
                activation_list.append(activation)

            delta_L = mean_squared_error_derivative(
                activation, training_example[1]) * self.activation_functions_derivatives[-1](z)

            delta = [np.zeros(self.dimensions[l]) for l in range(self.layers)]
            for l in range(self.layers - 1, 0, -1):

                if l == self.layers - 1:
                    delta[-1] = delta_L
                else:
                    delta[l] = np.dot(self.weights[l + 1].T, delta[l + 1]) * \
                        self.activation_functions_derivatives[l](z_list[l])

                # array[..., None] changes the row vector to a column vector and the right dimension
                # for calculation
                weight_delta_sum[l] = weight_delta_sum[l] + \
                    np.dot(delta[l][..., None],
                           activation_list[l - 1][..., None].T)

                bias_delta_sum[l] = bias_delta_sum[l] + delta[l]

        for l in range(1, self.layers):
            self.weights[l] = self.weights[l] - \
                weight_delta_sum[l] * self.eta / len(training_batch)
            self.biases[l] = self.biases[l] - \
                bias_delta_sum[l] * self.eta / len(training_batch)

    def print_network(self, layers = []):
        '''
For debugging purposes. Print the current weights and biases of the network to stdout. If layers is parsed print only those layers
'''
        if layers == []:
            for l in range(1, self.layers):
                print(f'Layer {l}: {self.weights[l]}; {self.biases[l]}')
        else:
            for l in layers:
                print(f'Layer {l}: {self.weights[l]}; {self.biases[l]}')



def main():
    '''
    NN = NeuralNetwork([2, 2, 1], 0.9, weights = [np.array([]), np.array([[0, 1], [1, 0]]), np.array([2, 3])],
            biases = [np.array([]), np.array([0]), np.array([0])],
            activation_functions = [0, lambda x: x), np.vectorize(lambda x: x)])
    #NN.initialize_network()
    print(NN.feed_forward(np.array([2, 1])))
    '''
    pass


if __name__ == '__main__':
    main()
