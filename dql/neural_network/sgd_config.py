# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
# Number of individuals in the population
import numpy as np

# dimensions of the Neural Network
DIMENSIONS = [1, 4, 1]

# learning rate eta
ETA = 0.1

# activation function names
ACTIVATION_FUNCTIONS = ["ReLU", "ReLU", "sigmoid"]

# start point of the function
START = 0

# end point of the function
STOP = np.pi

# functino to be approximated
def f(x):
    return (1 + np.cos(x)) / 2
