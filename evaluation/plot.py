# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import json
import math
import os
import random
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from dql.agent.agent import DQLAgent
from dql.agent.agent import ReplayMemory
from dql.environment.pong_env import PongEnvDQL
from dql.neural_network.neural_network import NeuralNetwork
from etc.activation_functions import *
from neat.genetics import *
from neat.neat import NEAT
from neat.pong_env import PongEnvNEAT

plt.style.use("ggplot")


def plot(fitness_hist, label):
    x = list(range(0, len(fitness_hist)))
    y = fitness_hist

    plt.plot(x, y, label=label)


def main():
    sys.setrecursionlimit(2**15)

    # NEAT

    N = NEAT(PongEnvNEAT, 20, (1, 1, 0.4), (0.8, 0.9),
             (0.02, 0.02), 0.1, 0.5, 10000)

    N.make_population_connected()

    #N.iterate(10, print_frequency=1)

    N.save_population()

    N.simulate_population(10000)

    # DQL
    env = PongEnvDQL()
    agt = DQLAgent(env, load_network_path='latest')

    agt.learn(20)

    # Plots

    plot(N.fitness_hist, "average")
    plot(N.best_fitness_hist, "best")

    plt.legend()
    plt.show()

    plot(N.generation_time_hist, "time")

    plt.legend()
    plt.show()

    plot(agt.fitness_hist, "fitness")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
