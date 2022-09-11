# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import argparse
import faulthandler
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
import dql.neural_network.sgd_config as SGD_cfg
import neat.config as NEAT_cfg
from dql.agent.agent import DQLAgent
from dql.agent.agent import ReplayMemory
from dql.environment.cartpole_gym_env import CartpoleEnvDQL
from dql.environment.pong_env import PongEnvDQL
from dql.neural_network.neural_network import NeuralNetwork
from dql.neural_network.sgd_test import sgd_main 
from etc.activation_functions import *
from neat.cartpole_gym_env import CartpoleEnvNEAT
from neat.genetics import *
from neat.neat import NEAT
from neat.pong_env import PongEnvNEAT

plt.style.use("ggplot")

#sys.stdout = open("traceback.txt", "w")


def trace(frame, event, arg):
    with open("traceback.txt", "a") as f:
        f.write("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

# sys.settrace(trace)


def plot(hist, label):
    x = list(range(0, len(hist)))
    y = hist

    plt.plot(x, y, label=label)


def args() -> 'argparse':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand", required=True,
                                       help='Specify whether NEAT, DQL or SGD should be run')

    # arguments for dql
    parser_dql = subparsers.add_parser('dql', help='Run DQL')
    parser_dql.add_argument('-e', '--episodes', dest='episodes', required=True,
                            help='Number of episodes', type=int)
    parser_dql.add_argument(
        '-p', '--plot', help="Plot the fitness and time at the end", action="store_true")
    parser_dql.add_argument(
        '-l', '--live-plot', help="Plot the fitness live", action="store_true")
    parser_dql.add_argument(
        '-s', '--save-data', help='Save all the information concerning the epoch',
        dest='save_data', action="store_true")
    parser_dql.add_argument('-g', '--game', help='Specify the game', required=True, dest='game',
                            choices=['pong', 'cartpole'], type=str)
    parser_dql.add_argument(
        '-r', '--render', help='Render the game', dest='render', action="store_true")
    parser_dql.add_argument(
        '-v', '--verbose', help='Print info', dest='verbose', action="store_true")
    parser_dql.add_argument(
        '--reward-system', help='How the (pong) enviroment should give out rewards', choices=['v0', 'v1', 'v2', 'v3'], type=str)

    # arguments for neat
    parser_neat = subparsers.add_parser('neat', help='Run NEAT')
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
    parser_neat.add_argument(
        '-ps', '--protect-species', help='Allow each species to have at least one offspring',
        dest='protect_species', action="store_true")
    parser_neat.add_argument(
        '--reward-system', help='How the (pong) enviroment should give out rewards', choices=['v0', 'v1', 'v2', 'v3'], type=str)

    # arguments for sgd 
    parser_sgd = subparsers.add_parser('sgd', help='Run SGD')
    parser_sgd.add_argument(
        '-v', '--verbose', help='Print info', dest='verbose', action="store_true")
    parser_sgd.add_argument('-e', '--epochs', required=True, 
                            help="Specify the number of epochs SGD should run", type=int)
    parser_sgd.add_argument('-b', '--batch-size', required=True, 
                            help="Specify the batch size of SGD", type=int)
    parser_sgd.add_argument('-p', '--plot', action="store_true",
                            help="Show a plot of the approximation at the end")
    parser_sgd.add_argument('-s', '--save', action="store_true",
                            help="Save the data")
    parser_sgd.add_argument('--save-network', action="store_true",
                            help="Save the Neural Network")

    return parser.parse_args()


def main():
    sys.setrecursionlimit(2**12)

    arguments = args()

    if not arguments.verbose:
        sys.stdout = open(os.devnull, "w")

    # DQL
    if arguments.subcommand == 'dql':
        games = {
            "pong": PongEnvDQL,
            "cartpole": CartpoleEnvDQL
        }
        env = games[arguments.game]

        agt = DQLAgent(
            env(alpha=DQL_cfg.ALPHA,
                render=arguments.render,
                reward_system=arguments.reward_system),
            DQL_cfg.MEMORY_SIZE,
            DQL_cfg.DISCOUNT_FACTOR,
            DQL_cfg.MINIBATCH_SIZE,
            DQL_cfg.LEARNING_RATE,
            DQL_cfg.EPS_DECAY,
            DQL_cfg.DONE_EPS,
            DQL_cfg.TARGET_NN_UPDATE_FREQ,
            DQL_cfg.LOAD_NETWORK_PATH,
            DQL_cfg.LIVE_PLOT_FREQ,
            arguments.live_plot,
            DQL_cfg.NN_SAVE_FREQ,
            save_data=arguments.save_data,
            game=arguments.game,
        )

        agt.learn(arguments.episodes)

        if arguments.plot:
            plot(agt.performance_hist, "fitness")

            plt.legend()
            plt.show()

            plot(agt.time_hist, "time")

            plt.legend()
            plt.show()

    # NEAT
    elif arguments.subcommand == 'neat':
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
            vary_delta_t=arguments.vary_delta_t,
            protect_species=arguments.protect_species,
            game=arguments.game,
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

    elif arguments.subcommand == "sgd":
        sgd_main(
            SGD_cfg.DIMENSIONS,
            SGD_cfg.ETA,
            SGD_cfg.ACTIVATION_FUNCTIONS,
            SGD_cfg.START,
            SGD_cfg.STOP,
            SGD_cfg.f,
            arguments.epochs,
            arguments.batch_size,
            arguments.save, 
            arguments.save_network, 
            arguments.plot, 
        ) 


if __name__ == '__main__':
    main()
    """
    with open("traceback.txt", "w") as f:
        faulthandler.enable(file=f)
        main()

    if faulthandler.is_enabled():
        faulthandler.diable()
    """
# Example use: python3 main.py dql -g cartpole -e 1000
