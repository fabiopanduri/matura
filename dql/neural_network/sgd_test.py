# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
"""
Used to test SGD performance
"""
import datetime
import json
import os
import sys
import time

import matplotlib.pyplot as plt

from dql.neural_network.neural_network import *


def sgd_main(dimensions,
             eta,
             activation_functions,
             START,
             STOP,
             f,
             epochs=100,
             batch_size=1000,
             save=False,
             save_net=False,
             plot=False,
             ):
    NN = NeuralNetwork(dimensions, eta,
                       activation_functions=activation_functions
                       )
    NN.initialize_network()

    print(
        f'[INFO] Starting SGD training with {epochs} epochs and a batch size of {batch_size}')

    # generate random training batch
    train = []
    for _ in range(batch_size):
        x = np.random.uniform(START, STOP)
        train.append((np.array([x]), np.array([f(x)])))

    error = []
    time_history = []
    for i in range(epochs):
        if i % int(epochs / 10) == 0:
            print(f'[INFO] Epoch: {i}/{epochs}', end='\r')

        # perform the gradient descent step
        t_0 = time.perf_counter()
        NN.stochastic_gradient_descent(train)
        t = time.perf_counter() - t_0
        time_history.append(t)

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

    # make a plot depicting the approximations of the neural network
    predict = []
    for x in np.arange(START, STOP, 0.01):
        predict.append(NN.feed_forward(np.array([x])))

    # get the data of the actual function
    actual = [f(x) for x in np.arange(START, STOP, 0.01)]

    if plot:
        plt.figure()
        plt.plot(np.arange(START, STOP, 0.01), predict, label="NN prediction")
        plt.plot(np.arange(START, STOP, 0.01), actual, label="Target function")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")

        plt.figure()
        plt.plot(list(range(0, len(error))), error)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.figure()
        plt.plot(list(range(0, len(time_history))), time_history)
        plt.xlabel("Epochs")
        plt.ylabel("Time in s")
        plt.show()

    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # save the neural network to
    if save_net:
        path = f'SGD_nets_saves/SGD_net_{time_stamp}.npz'
        NN.save_network(path)

    if save:
        data = {
            "time": time_stamp,
            "prediction": list(map(float, predict)),
            "loss": list(map(float, error)),
            "time history": time_history,
            "hyperparameters": {
                "epochs": epochs,
                "batch size": batch_size,
                "dimensions": dimensions,
                "learnig rate eta": eta,
                "activation function names": activation_functions,
            }
        }

        with open(f"SGD_saves/SGD_{time_stamp}.json", "w") as file:
            file.write(json.dumps(data, indent=4))
