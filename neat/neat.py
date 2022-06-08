# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import math

import numpy as np

from etc.activation_functions import *
from neat.genetics import *
from neat.pong_env import PongEnv


class NEAT:
    """
    This class contains the genetic algorithm part of NEAT
    It is only used for training
    """

    def __init__(
        self,
        env,
        population_size,
        speciation_constants,
        weight_mutation_constants,
        node_connection_mutation_constants,
        delta_t,
        r,
        simulation_time,
    ):
        self.env = env
        temp_env = env()
        self.nn_base_dimensions = temp_env.nn_base_dimensions()
        self.population_size = population_size
        self.speciation_constants = speciation_constants
        self.weight_mutation_constants = weight_mutation_constants
        self.node_connection_mutation_constants = node_connection_mutation_constants
        self.delta_t = delta_t
        self.r = r
        self.simulation_time = simulation_time

        self.population = []
        self.global_connection_innovation_number = self.nn_base_dimensions[0] * \
            self.nn_base_dimensions[1]
        self.global_node_innovation_number = sum(self.nn_base_dimensions)
        self.species = {}

    def iterate(self, generations, print_frequency=1):
        """
        Function that runs iterations of NEAT
        """

        for i in range(generations):
            self.simulate_population(self.simulation_time)

            if i % print_frequency == 0:
                best = sorted(self.population,
                              key=lambda x: x.fitness)[-1].fitness
                print(f'[INFO] Generation {i} done. Best = {best}')

            self.speciation()

            self.adjust_population_fitness()

            new_N = self.get_species_sizes()

            self.mate(new_N)

    def simulate_population(self, max_T):
        """
        Method to simulate the interaction of the agent with the environment for every individual in the 
        population.
        This is done until a terminal state is reached or we have more than max_T steps
        """

        for individual in self.population:
            env = self.env()

            state_0 = env.make_observation()
            action = individual.feed_forward(state_0)
            for t in range(max_T):
                state, reward, terminated = env.step(action)

                if terminated:
                    # use a weighted reward depending on when the terminal state is reached
                    if reward > 0:
                        individual.fitness = (1 - (t / max_T)) * reward
                    else:
                        individual.fitness = ((t / max_T) - 2) * abs(reward)
                    t = max_T
                    break

                prediction = individual.feed_forward(state)
                action_i = np.argmax(np.array(prediction))

                action = env.possible_actions[action_i]
                t += 1

            else:
                individual.fitness = 0

    def speciation(self):
        """
        This method will create species for the current generation based of the species of the last
        generation
        If an individual does not fit into any species a new species is created
        """

        species = {}
        for individual in self.population:
            for i, representatives in self.species.items():
                if random.choice(representatives).delta(individual) < self.delta_t:
                    if i in species:
                        species[i].append(individual)
                    else:
                        species[i] = [individual]
                    break

            else:
                if not species.keys() and not self.species.keys():
                    new_species_i = 0
                elif not species.keys():
                    new_species_i = max(self.species.keys()) + 1
                elif not self.species.keys():
                    new_species_i = max(species.keys()) + 1
                else:
                    new_species_i = max(
                        max(species.keys()),
                        max(self.species.keys())
                    ) + 1
                species[new_species_i] = [individual]

        self.species = species

    def adjust_population_fitness(self):
        """
        Adjust the fitness of all individuals in the population
        """

        for s in self.species.values():
            for individual in s:
                individual.adjust_fitness(s)

    def get_species_sizes(self):
        """
        Calculate the number of offspring each species can produce 
        """

        s_fitness = {i: 0 for i in self.species.keys()}
        for s_index, s in self.species.items():
            for individual in s:
                s_fitness[s_index] += individual.fitness

            s_fitness[s_index] /= len(s)

        mean_adjusted_fitness = sum(s_fitness.values())

        base = {i: (s_f / mean_adjusted_fitness) *
                self.population_size for i, s_f in s_fitness.items()}
        base_int = {i: math.floor(b) for i, b in base.items()}
        base_mod_1 = sorted([(i, b % 1)
                            for i, b in base.items()], key=lambda x: x[1])

        base_offspring = sum(base_int.values())
        additional_offspring = self.population_size - base_offspring

        for additional in range(additional_offspring):
            s = base_mod_1[additional % len(self.species)]
            base_int[s[0]] += 1

        return base_int

    def mate(self, new_N):
        """
        This method performs the mating step and generates a new generation
        """

        new_generation = []

        for s_index, s in self.species.items():

            sorted_s = sorted(s, key=lambda x: - x.fitness)
            l = max(math.ceil(len(sorted_s) * self.r), 1)
            mating_s = sorted_s[0:l]
            N = new_N[s_index]

            for i in range(N):
                if len(mating_s) == 1:
                    new_generation.append(mating_s[0])
                    continue

                p1, p2 = random.sample(mating_s, k=2)
                child = Genome.crossover(p1, p2)

                new_generation.append(child)

        self.population = new_generation

    def mutate(self):
        """
        Mutate every individual in the population
        """

        for individual in self.population:
            individual.mutate_weights(
                weight_mutation_constants=self.weight_mutation_constants)

            if random.random() < self.node_connection_mutation_constants[0]:
                individual.add_node(
                    self.global_node_innovation_number,
                    self.global_connection_innovation_number
                )

                self.global_node_innovation_number += 1
                self.global_connection_innovation_number += 1

            if random.random() < self.node_connection_mutation_constants[1]:
                individual.add_connection(
                    self.global_connection_innovation_number
                )

                self.global_connection_innovation_number += 1

    def make_population_empty(self, activation_functions_hidden=[], activation_functions_output=[]):
        """
        Makes a generation of the given population size with the given nn_base_dimensions
        The nets in the population are connection free, so they only contain nodes
        """

        self.population = []

        for i in range(self.population_size):
            self.population.append(
                Genome.make_empty_genome(
                    self.nn_base_dimensions[0],
                    sum(self.nn_base_dimensions[1:-1]),
                    self.nn_base_dimensions[-1],
                    activation_functions_hidden,
                    activation_functions_output
                )
            )

    def make_population_connected(self, activation_functions=[]):
        """
        Makes a generation of the given population size with the given nn_base_dimensions
        The nets in the population are all fully-connected digraphs from the input nodes to
        the output nodes
        """

        self.population = []

        for i in range(self.population_size):
            self.population.append(
                Genome.make_connected_genome(
                    self.nn_base_dimensions[0],
                    self.nn_base_dimensions[-1],
                    activation_functions
                )
            )


def main():
    N = NEAT(PongEnv, 10, (1, 1, 1), (0.8, 0.9), (0.1, 0.1), 1, 0.5, 100000)

    N.make_population_connected()

    N.iterate(1000, print_frequency=50)

    N.simulate_population(10000)

    for individual in N.population:
        print(individual.fitness)

    """
    s1 = N.speciation()
    s2 = N.speciation()
    N.adjust_population_fitness()

    print("")
    for individual in N.population:
        print(individual.fitness)
        """

    """
    s1 = N.speciation({})
    print(s1)
    s2 = N.speciation(s1)
    print(s2)
    """

    """
    for p in N.population:
        p.draw()
    """


if __name__ == "__main__":
    main()
