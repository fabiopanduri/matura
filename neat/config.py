# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
# Number of individuals in the population
"""
NEAT config
"""

POPULATION_SIZE = 50

# Constants for speciation, i.e., c_1, c_2, c_3
SPECIATION_CONSTANTS = (2, 2, 0.4)

# Threshold determining the distance allowed for an individual to be part of a species
DELTA_T = 1.75

# Constants that determine the probability of weight mutation
WEIGHT_MUTATION_CONSTANTS = (0.8, 0.9)

# Constants that determine the probabilities of node and connection mutations
# First constant is the probability a new connection is added to an individual
# The second constant is the probability a new node is added to an individual
NODE_CONNECTION_MUTATION_CONSTANTS = (0.001, 0.003)

# Constant specifying the probability an inherited connection is disabled if either parent had it
# disabled
CONNECTION_DISABLE_CONSTANT = 0.75

# Percentage of each Species that is allowed to have offspring
R = 0.5

# Maximal number of frames a game is simulated
SIMULATION_TIME = 1000

#
ALPHA = 2000
