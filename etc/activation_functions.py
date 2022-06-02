# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLU(x):
    return max(0, x)


def ReLU_derivative(x):
    return 1 if x > 0 else 0


def linear(x):
    return x


def linear_derivative(x):
    return 1


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    return (1 - tanh(x)**2)


def PReLU(x, alpha):
    return alpha * x if x < 0 else x


def PReLU_derivative(x, alpha):
    return alpha if x < 0 else 1


activation_functions = {
    "sigmoid": sigmoid,
    "ReLU": ReLU,
    "linear": linear,
    "tanh": tanh,
    "PReLU": PReLU,
}

activation_functions_derivatives = {
    "sigmoid": sigmoid_derivative,
    "ReLU": ReLU_derivative,
    "linear": linear_derivative,
    "tanh": tanh_derivative,
    "PReLU": PReLU_derivative,
}
