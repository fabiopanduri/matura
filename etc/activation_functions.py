# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from typing import Dict, Callable

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
        return max(0, x)

def ReLU_derivative(x):
        return 1 if x > 0 else 0


activation_functions: Dict[str, Callable] = {
	'sigmoid' : sigmoid,
	'ReLU' : ReLU,
}

activation_functions_derivatives: Dict[str, Callable] = {
	'sigmoid' : sigmoid_derivative,
	'ReLU' : ReLU_derivative,
}
