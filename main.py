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

import dql.config as DQL_cfg
import neat.config as NEAT_cfg
from dql.agent.agent import DQLAgent
from dql.agent.agent import ReplayMemory
from dql.environment.cartpole_gym_env import CartpoleEnvDQL
from dql.environment.pong_env import PongEnvDQL
from dql.neural_network.neural_network import NeuralNetwork
from etc.activation_functions import *
from neat.cartpole_gym_env import CartpoleEnvNEAT
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
        '-p', '--plot', help="Plot the fitness and time at the end", action="store_true")
    parser_dql.add_argument(
        '-l', '--live-plot', help="Plot the fitness live", action="store_true")
    parser_dql.add_argument('-g', '--game', help='Specify the game', required=True, dest='game',
                            choices=['pong', 'cartpole'], type=str)
    parser_dql.add_argument(
        '-r', '--render', help='Render the game', dest='render', action="store_true")
    parser_dql.add_argument(
        '-v', '--verbose', help='Print info', dest='verbose', action="store_true")

    # arguments for neat
    parser_neat = subparsers.add_parser('neat', help='Test NEAT')
    parser_neat.add_argument('-i', '--iterations', required=True,
                             help='Number of iterations', type=int)
    parser_neat.add_argument(
        '-p', '--plot', help="Plot the fitness and time at the end", action="store_true")
    parser_neat.add_argument(
        '-la', '--live-average', help="Plot the average fitness live", action="store_true")
    parser_neat.add_argument(
        '-lb', '--live-best', help="Plot the best fitness live", action="store_true")
    parser_neat.add_argument(
        '-lt', '--live-time', help="Plot the time live", action="store_true")
    parser_neat.add_argument(
        '-s', '--save-data', help='Save all the information concerning the iteration',
        dest='save_data', action="store_true")
    parser_neat.add_argument('-g', '--game', help='Specify the game', required=True, dest='game',
                             choices=['pong', 'cartpole'], type=str)
    parser_neat.add_argument(
        '-r', '--render', help='Render the game', dest='render', action="store_true")
    parser_neat.add_argument(
        '-v', '--verbose', help='Print info', dest='verbose', action="store_true")
    parser_neat.add_argument(
        '-c', '--connected', help='Start with a connected graph', dest='connected', action="store_true")
    parser_neat.add_argument(
        '-vd', '--vary-delta-t', help='Vary the delta-t value', dest='vary_delta_t', action="store_true")

    return parser.parse_args()


def main():
    sys.setrecursionlimit(2**15)

    arguments = args()

    if not arguments.verbose:
        sys.stdout = open(os.devnull, "w")

    # NEAT
    if arguments.subcommand == 'neat':
        games = {
            "pong": PongEnvNEAT,
            "cartpole": CartpoleEnvNEAT
        }
        env = games[arguments.game]

        N = NEAT(
            env,
            NEAT_cfg.POPULATION_SIZE,
            NEAT_cfg.SPECIATION_CONSTANTS,
            NEAT_cfg.WEIGHT_MUTATION_CONSTANTS,
            NEAT_cfg.NODE_CONNECTION_MUTATION_CONSTANTS,
            NEAT_cfg.DELTA_T,
            NEAT_cfg.R,
            NEAT_cfg.SIMULATION_TIME,
            NEAT_cfg.CONNECTION_DISABLE_CONSTANT,
            NEAT_cfg.ALPHA,
            render=arguments.render,
            vary_delta_t=arguments.vary_delta_t
        )

        if arguments.connected:
            N.make_population_connected()
        else:
            N.make_population_empty()

        N.iterate(
            arguments.iterations,
            print_frequency=1,
            live_f=arguments.live_average,
            live_b=arguments.live_best,
            live_t=arguments.live_time,
            save_data=arguments.save_data,
        )

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
            "cartpole": CartpoleEnvDQL
        }
        env = games[arguments.game]

        agt = DQLAgent(
            env(alpha=DQL_cfg.ALPHA,
                render=arguments.render),
            DQL_cfg.MEMORY_SIZE,
            DQL_cfg.DISCOUNT_FACTOR,
            DQL_cfg.MINIBATCH_SIZE,
            DQL_cfg.LEARNING_RATE,
            DQL_cfg.EPS_DECAY,
            DQL_cfg.DONE_EPS,
            DQL_cfg.TARGET_NN_UPDATE_FREQ,
            DQL_cfg.LOAD_NETWORK_PATH,
            arguments.live_plot,
            DQL_cfg.LIVE_PLOT_FREQ,
            DQL_cfg.NN_SAVE_FREQ,
        )

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

# Example use: py -m eval.plot dql -g cartpole -e 1000 -l
