# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
"""
DQL Config
"""
# Size of the replay memory
MEMORY_SIZE = 2000

# Discount factor (gamma) for the Q-function
DISCOUNT_FACTOR = 0.95

# Size of the minibatches that are parsed to SGD
MINIBATCH_SIZE = 32

# Learning rate (eta) of SGD
LEARNING_RATE = 0.001

# Epsilon (exploration rate) exponential decay rate
EPS_ANNEALING_RATE = 0.995

# Terminal epsilon
DONE_EPS = 0.1

# After how many steps the target Q-Network should be updated
TARGET_NN_UPDATE_FREQ = 100

# File path of NN file to be used instead of randomly initialized NN
LOAD_NETWORK_PATH = None

# How often (in episodes) the live plot should be updated
LIVE_PLOT_FREQ = 1

# How often (steps) to save the NN
NN_SAVE_FREQ = 100000

# Fitness parameter
ALPHA = 100

# After how many steps an episode terminates at the latest
MAX_SIMULATION_TIME = 10000
