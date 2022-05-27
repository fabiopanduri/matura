# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.


from genetics import *
from activation_functions import *


class NEAT:
    """
    This class contains the genetic algorithm part of NEAT
    It is only used for training
    """

    def __init__(
        self,
        population_size,
        nn_base_dimensions,
    ):
        self.population_size = population_size
        self.nn_dimensions = nn_base_dimensions
        self.global_innovation_number = 0

    def make_population(self):
        """
        Makes a generation of the given population size with the given nn_base_dimensions
        The nets in the population are connection free, so they only contain nodes
        """

        self.population = []

        for i in range(self.population_size):
            self.population.append(
                Genome.empty_genome(
                    self.nn_dimensions[0],
                    sum(self.nn_dimensions[1:-1]),
                    self.nn_dimensions[-1],
                )
            )

    def train(self, T):
        pass


def main():
    """
    N = NEAT(1, [4, 3, 2, 1, 4])
    N.make_population()
    for p in N.population:
            p.draw()
    """


if __name__ == "__main__":
    main()
