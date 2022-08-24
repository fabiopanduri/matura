# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import json
import math
import os
import random
import time

import numpy as np
import pygame

from etc.activation_functions import *
from neat.cartpole_gym_env import CartpoleEnvNEAT
from neat.genetics import *
from neat.neat import NEAT
from neat.pong_env import PongEnvNEAT


def main():
    iterations = 10
    best_p = 0.2
    print_frequency = 1
    FPS_LIMIT = 60

    N = NEAT(env=CartpoleEnvNEAT,
             population_size=20,
             speciation_constants=(1, 1, 0.4),
             weight_mutation_constants=(0.8, 0.9),
             node_connection_mutation_constants=(0.02, 0.02),
             delta_t=0.1,
             r=0.5,
             simulation_time=10000,
             connection_disable_constant=0.1,
             alpha=100,
             render=True,
             )

    N.make_population_connected()
    # N.population = NEAT.load_population()

    print(f"[INFO] Choosing the best individuals")
    best = []
    for i in range(iterations):
        if i % print_frequency == 0:
            print(f"[INFO] Iteration {i}.")

        N.simulate_population(10000)
        best = best + sorted(N.population, key=lambda x: -
                             x.fitness)[0: max(1, int(len(N.population) * best_p))]
        N.population = best

    print("")

    chosen = random.choice(best)

    env = PongEnvNEAT()
    clock = pygame.time.Clock()
    state = env.make_observation()
    action = chosen.feed_forward(state)
    i = 0
    while True:
        # print("Step: ", env.step(['stay']), "Score: ", env.score)
        clock.tick(FPS_LIMIT)

        state, _, _ = env.step(action)

        prediction = chosen.feed_forward(state)
        action_i = np.argmax(np.array(prediction))

        action = env.possible_actions[action_i]

        if i % 100 == 0:
            print(f"[INFO] {i=} {prediction=} {action=}")
        i += 1


if __name__ == "__main__":
    main()
