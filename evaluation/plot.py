# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import argparse
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
from dql.environment.cartpole_gym_env import CartpoleEnvDQL
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


def args() -> 'argparse':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand", required=True,
                                       help='Specify whether NEAT or DQL should be tested ')

    # arguments for dql
    parser_dql = subparsers.add_parser('dql', help='Test DQL')
    parser_dql.add_argument('-e', '--episodes', dest='episodes', required=True,
                            help='Number of episodes', type=int)
    parser_dql.add_argument(
        '-p', '--plot', help="Plot the fitness", action="store_true")

    # arguments for neat
    parser_neat = subparsers.add_parser('neat', help='Test NEAT')
    parser_neat.add_argument('-i', '--iterations', required=True,
                             help='Number of iterations', type=int)
    parser_neat.add_argument(
        '-p', '--plot', help="Plot the fitness", action="store_true")

    # general
    parser.add_argument('-g', '--game', help='Specify the game', required=True, dest='game',
                        choices=['pong', 'cart-pole'], type=str)

    return parser.parse_args()


def main():
    sys.setrecursionlimit(2**15)

    arguments = args()

    # NEAT
    if arguments.subcommand == 'neat':
        games = {
            "pong": PongEnvNEAT,
        }
        env = games[arguments.game]

        N = NEAT(env, 20, (1, 1, 0.4), (0.8, 0.9),
                 (0.02, 0.02), 0.1, 0.5, 10000)

        N.make_population_connected()

        N.iterate(arguments.iterations, print_frequency=1)

        N.save_population()

        N.simulate_population(10000)

        if arguments.plot:
            plot(N.fitness_hist, "average")
            plot(N.best_fitness_hist, "best")

            plt.legend()
            plt.show()

            plot(N.generation_time_hist, "time")

            plt.legend()
            plt.show()

    # DQL
    if arguments.subcommand == 'dql':
        games = {
            "pong": PongEnvDQL,
            "cart-pole": CartpoleEnvDQL
        }
        env = games[arguments.game]

        agt = DQLAgent(env(), load_network_path='latest')

        agt.learn(arguments.episodes)

        if arguments.plot:
            plot(agt.fitness_hist, "fitness")

            plt.legend()
            plt.show()

            plot(agt.time_hist, "time")

            plt.legend()
            plt.show()


if __name__ == '__main__':
    main()
