# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
from etc.activation_functions import *
from neat.genetics import *


class NEAT:
    """
    This class contains the genetic algorithm part of NEAT
    It is only used for training
    """

    def __init__(
        self,
        env,
        population_size,
        nn_base_dimensions,
        speciation_constants,
        weight_mutation_constants,
        delta_t
    ):
        self.env = env
        self.population_size = population_size
        self.population = []
        self.nn_dimensions = nn_base_dimensions
        self.global_innovation_number = 0
        self.speciation_constants = speciation_constants
        self.weight_mutation_constants = weight_mutation_constants
        self.delta_t = delta_t

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
            for t in range(max_t):
                state, reward = env.step(action)

                if env.is_terminal():
                    # use a weighted reward depending on when the terminal state is reached
                    individual.fitness = (1 - (t / max_T)) * reward
                    t = max_T
                    break

                action = individual.feed_forward(state)
                t += 1

            else:
                individual.fitness = 0

    def speciation(self, previous_gen_species):
        """
        This method will create species for the current generation based of the species of the last
        generation
        If an individual does not fit into any species a new species is created
        """

        species = {}
        for individual in self.population:
            for i, representatives in previous_gen_species.items():
                if random.choice(representatives).delta(individual) < self.delta_t:
                    if i in species:
                        species[i].append(individual)
                    else:
                        species[i] = [individual]
                    break

            else:
                if not species.keys() and not previous_gen_species.keys():
                    new_species_i = 0
                elif not species.keys():
                    new_species_i = max(previous_gen_species.keys()) + 1
                elif not previous_gen_species.keys():
                    new_species_i = max(species.keys()) + 1
                else:
                    new_species_i = max(
                        max(species.keys()),
                        max(previous_gen_species.keys())
                    ) + 1
                species[new_species_i] = [individual]

        return species

    def get_species_sizes(self, species):
        """
        Calculate the number of offspring each species can produce 
        """

        s_fitness = {i: 0 for i in species.keys()}
        for s_index, s in species:
            for individual in s:
                s_fitness[s_index] += individual.fitness

        total_fitness = sum(s_fitness.values())

        return {i: s_f // total_fitness for i, s_f in s_fitness.items()}

    def make_population_empty(self, activation_functions_hidden=[], activation_functions_output=[]):
        """
        Makes a generation of the given population size with the given nn_base_dimensions
        The nets in the population are connection free, so they only contain nodes
        """

        self.population = []

        for i in range(self.population_size):
            self.population.append(
                Genome.make_empty_genome(
                    self.nn_dimensions[0],
                    sum(self.nn_dimensions[1:-1]),
                    self.nn_dimensions[-1],
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
                    self.nn_dimensions[0],
                    self.nn_dimensions[-1],
                    activation_functions
                )
            )


def main():
    N = NEAT(int, 4, [4, 4], (1, 1, 1), (0.8, 0.9), 1)
    N.make_population_connected()

    s1 = N.speciation({})
    print(s1)
    s2 = N.speciation(s1)
    print(s2)

    """
    for p in N.population:
        p.draw()
    """


if __name__ == "__main__":
    main()
