# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt

from dql.neural_network.neural_network import *


def f(x) -> float:
    return (np.cos(x) + 1) / 2


def main():
    '''
    NN = NeuralNetwork([1, 4, 1], 0.1,
                    activation_functions = ['ReLU', 'ReLU', 'sigmoid'],
                    )
    '''

    NN = NeuralNetwork.load_network('sgd_test.npz')

    START = 0
    STOP = np.pi

    try:
        epochs = int(sys.argv[1])
    except IndexError:
        epochs = 100

    try:
        batch_size = int(sys.argv[2])
    except IndexError:
        batch_size = 1000

    print(
        f'[INFO] Starting SGD training with {epochs} epochs and a batch size of {batch_size}')

    # generate random training batch
    train = []
    for _ in range(batch_size):
        x = np.random.uniform(START, STOP)
        train.append((np.array([x]), np.array([f(x)])))

    error = []
    for i in range(epochs):
        if i % int(epochs / 10) == 0:
            print(f'[INFO] Epoch: {i}/{epochs}', end='\r')

        # perform the gradient descent step
        NN.stochastic_gradient_descent(train)

        # test the neural network on some newly generated data
        test = []
        for _ in range(10):
            x = np.random.uniform(START, STOP)
            test.append((np.array([x]), np.array([f(x)])))

        sum = 0
        for t in test:
            prediction = NN.feed_forward(t[0])
            sum += 0.5 * (prediction - t[1])**2

        error.append(sum / len(test))

    print(f'[INFO] Epoch: {epochs}/{epochs}.')

    # save the neural network to
    NN.save_network('sgd_test.npz')

    # make a plot depicting the approximations of the neural network
    predict = []
    for x in np.arange(START, STOP, 0.01):
        predict.append(NN.feed_forward(np.array([x])))

    # get the data of the actual function
    actual = [f(x) for x in np.arange(START, STOP, 0.01)]

    plt.figure()
    plt.plot(np.arange(START, STOP, 0.01), predict)
    plt.plot(np.arange(START, STOP, 0.01), actual)
    plt.figure()
    plt.plot(list(range(0, len(error))), error)
    plt.show()


if __name__ == '__main__':
    main()
