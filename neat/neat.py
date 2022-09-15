# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import json
import math
import os
import time

from etc.activation_functions import *
from neat.genetics import *


INX = 0


class NEAT:
    INX = 0
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
        connection_disable_constant,
        alpha,
        optimization="max",
        render=False,
        vary_delta_t=False,
        protect_species=True,
        game='',
        old_speciation=False,
        reward_system="v0",
    ):
        self.env = env
        temp_env = env(simulation_time)
        self.nn_base_dimensions = temp_env.nn_base_dimensions()
        self.population_size = population_size
        self.speciation_constants = speciation_constants
        self.weight_mutation_constants = weight_mutation_constants
        self.node_connection_mutation_constants = node_connection_mutation_constants
        self.base_delta_t = delta_t
        self.vary_delta_t = vary_delta_t
        self.r = r
        self.simulation_time = simulation_time
        self.connection_disable_constant = connection_disable_constant
        self.alpha = alpha
        self.optimization = optimization
        self.protect_species = protect_species
        self.game = game
        self.old_speciation = old_speciation
        self.reward_system = reward_system

        self.population = []
        self.global_connection_innovation_number = self.nn_base_dimensions[0] * \
            self.nn_base_dimensions[1]
        self.global_node_innovation_number = sum(self.nn_base_dimensions)
        self.species = []

        self.species_size_hist = [0]
        self.fitness_hist = []
        self.best_fitness_hist = []
        self.generation_time_hist = []

        self.render = render

    def iterate(self, generations, print_frequency=1, save_frequency=10, live_f=False, live_b=False,
                live_t=False, save_data=False):
        """
        Function that runs iterations of NEAT
        """

        t_0 = time.perf_counter()
        for i in range(generations):
            self.simulate_population(self.simulation_time) 

            t = time.perf_counter() - t_0

            self.generation_time_hist.append(t)

            sorted_pop = sorted(self.population, key=lambda x: x.fitness)
            average = sum(map(lambda x: x.fitness, sorted_pop)
                          ) / len(sorted_pop)
            best = sorted_pop[-1].fitness

            self.fitness_hist.append(average)
            self.best_fitness_hist.append(best)

            #sorted_pop[-1].draw()

            if live_f:
                self.plot_fitness_live()

            if live_b:
                self.plot_best_fitness_live()

            if live_t:
                self.plot_time_live()

            if i % print_frequency == 0:
                print(
                    f'[INFO] Generation {i} done. Took {t:.2f} s')
                print(
                    f'[INFO] Best = {best:.2f}. Average = {average:.2f}')
                print(
                    f'[INFO] Species: {len(self.species)}. Population Count: {len(self.population)}')
                print(
                    f'[INFO] Innovation Nodes, Connections: {self.global_node_innovation_number}, {self.global_connection_innovation_number}'
                )
                print(
                    f'[INFO] Delta t: {self.delta_t(i/generations)}\n'
                )

            if i % save_frequency == 0:
                self.save_population()

            t_0 = time.perf_counter()

            if self.old_speciation:
                self.speciation(i/generations)
            else:
                self.speciation_old_free(i/generations)

            self.adjust_population_fitness()

            new_N = self.get_species_sizes(protect=self.protect_species)

            self.mate(new_N)

        if save_data:
            data = {
                "time": datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                "generations": generations,
                "fitness history": self.fitness_hist,
                "best fitness history": self.best_fitness_hist,
                "generation time history": self.generation_time_hist,
                "species size history": self.species_size_hist,
                "innovation numbers (nodes/connection)":
                    (self.global_node_innovation_number,
                     self.global_connection_innovation_number),
                "hyperparameters": {
                    "neural network base dimensions": self.nn_base_dimensions,
                    "population size": self.population_size,
                    "speciation constants (c_1, c_2, c_3)": self.speciation_constants,
                    "weight mutation constants": self.weight_mutation_constants,
                    "node connection mutation constants": self.node_connection_mutation_constants,
                    "connection disable constant": self.connection_disable_constant,
                    "base delta t": self.base_delta_t,
                    "vary delta t": self.vary_delta_t,
                    "r": self.r,
                    "simulation time": self.simulation_time,
                    "alpha": self.alpha,
                    "optimization": self.optimization,
                    "render": self.render
                },
                "game": self.game
            }

            file_name = (
                f"NEAT-iteration-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
            )

            with open(f"NEAT_iterations_saves/{file_name}", "w") as f:
                f.write(json.dumps(data, indent=4))

            print(
                f"[INFO] Saved iteration to NEAT_iterations_saves/{file_name}.")

    def simulate_population(self, max_t):
        """
        Method to simulate the interaction of the agent with the environment for every individual in the 
        population.
        This is done until a terminal state is reached or we have more than max_t steps
        """

        for individual in self.population:
            env = self.env(max_t=max_t, render=self.render, reward_system=self.reward_system)

            state_0 = env.make_observation()
            prediction = individual.feed_forward(state_0)
            action_i = np.argmax(np.array(prediction))

            action = env.possible_actions[action_i]
            for t in range(max_t):
                state, reward, terminated = env.step(action, t)

                if terminated:
                    individual.fitness = env.fitness(
                        t, reward, alpha=self.alpha)
                    break

                prediction = individual.feed_forward(state)
                action_i = np.argmax(np.array(prediction))

                action = env.possible_actions[action_i]
                t += 1

            else:
                individual.fitness = env.done_fitness

    def delta_t(self, p_t):
        if self.vary_delta_t and p_t > 0.1:
            species_size_last_gen = self.species_size_hist[-1]
            p = (species_size_last_gen / self.population_size)
            return self.base_delta_t * p
        else:
            return self.base_delta_t

    def speciation_old_free(self, i):
        """
        This method will create species for the current generation 
        If an individual does not fit into any species a new species is created
        Else it is placed into the species it fits in to
        """

        species = []
        for individual in self.population:
            for i, s in enumerate(species.copy()):
                # individual fits into a species
                if random.choice(s).delta(individual) < self.delta_t(i):
                    species[i].append(individual)
                    break

            # individual does not fit into a species
            else:
                species.append([individual])

        self.species = [s for s in species if s]
        self.species_size_hist.append(len(self.species))

    def speciation(self, i):
        """
        This method will create species for the current generation based of the species of the last
        generation
        If an individual does not fit into any species a new species is created
        """

        species = [[] for _ in self.species]
        representatives = [random.choice(s) for s in self.species]
        for individual in self.population:
            for i, r in enumerate(representatives):
                # print(r.delta(individual))
                # individual fits into a species
                if r.delta(individual) < self.delta_t(i):
                    species[i].append(individual)
                    break

            # individual does not fit into a species
            else:
                species.append([individual])

        self.species = [s for s in species if s]
        self.species_size_hist.append(len(self.species))

    def adjust_population_fitness(self):
        """
        Adjust the fitness of all individuals in the population
        """

        for s in self.species:
            for individual in s:
                individual.adjust_fitness(s)

    def get_species_sizes(self, protect=True):
        """
        Calculate the number of offspring each species can produce 
        """

        s_fitness = [0 for _ in range(len(self.species))]

        # get the average fitness of any species
        for s_index, s in enumerate(self.species):
            for individual in s:
                s_fitness[s_index] += individual.fitness

            s_fitness[s_index] /= len(s)

        mean_adjusted_fitness = sum(s_fitness)

        # float values for offspring numbers
        if protect:
            # every species is allowed to have at least one offspring
            base = [1 + (s_f / mean_adjusted_fitness) *
                    (self.population_size - len(self.species)) for s_f in s_fitness]
        else:
            base = [(s_f / mean_adjusted_fitness) *
                    self.population_size for s_f in s_fitness]

        # rounded down offspring numbers
        base_int = [math.floor(b) for b in base]

        # the offspring numbers mod 1 -> the species with the higher number get more additional
        # offspring depending on how much more are needed to fill the population
        base_mod_1 = sorted([(i, b % 1)
                            for i, b in enumerate(base)], key=lambda x: x[1])

        base_offspring = sum(base_int)
        additional_offspring = self.population_size - base_offspring

        for additional in range(additional_offspring):
            s = base_mod_1[additional % len(self.species)]
            base_int[s[0]] += 1

        """
        print(f"{base=}")
        print(f"{base_int=}")
        print(f"{base_mod_1=}")
        """

        return base_int

    def has_cycle(self, genome, node):
        if node in self.finished:
            return False
        if node in self.visited:
            return True

        self.visited.add(node)

        for connection in genome.connections:
            if connection.out_node_id == node: 
                if self.has_cycle(genome, connection.in_node_id):
                    return True

        self.finished.add(node)

    def mate(self, new_N, optimization="max"):
        """
        This method performs the mating step and generates a new generation
        """

        new_generation = []
        best = []

        for s_index, s in enumerate(self.species):

            # best individual to worst individual of the species
            sorted_s = sorted(s, key=lambda x: - x.fitness)

            # index such that the best r% of the species are chosen
            l = max(math.ceil(len(sorted_s) * self.r), 1)
            mating_s = sorted_s[0:l]

            N = new_N[s_index]
    
            i = 0
            while i < N:
                # the best performing individual is always passed on to the new generation
                if i == 0:
                    best.append(mating_s[0])
                    i += 1
                    continue

                elif len(mating_s) == 1:
                    c = Genome.load_network_from_raw_data(
                        mating_s[0].save_network_raw_data())
                    new_generation.append(c)
                    i += 1
                    continue

                p1, p2 = random.sample(mating_s, k=2)
                child = Genome.crossover(
                    p1, p2, self.connection_disable_constant)

                # checking for a cycle that could have been created in
                # crossover 
                cycle = False
                if child.connections:
                    for node in child.nodes.keys():
                        self.visited = set() 
                        self.finished = set() 
                        if self.has_cycle(child, node):
                            cycle = True
                            break
                    if not cycle:
                        new_generation.append(child)
                        i += 1
                else:
                    new_generation.append(child)
                    i += 1

        # all children are mutated, the best per species are not
        new_generation = self.mutate_offspring(new_generation)

        self.population = new_generation + best

    def mutate_offspring(self, offspring):
        """
        Mutate every individual in the population
        """

        for individual in offspring:
            individual.mutate_weights(
                weight_mutation_constants=self.weight_mutation_constants)

            # add a node
            if random.random() < self.node_connection_mutation_constants[0]:
                individual.add_node(
                    self.global_node_innovation_number,
                    self.global_connection_innovation_number
                )

                self.global_node_innovation_number += 1
                self.global_connection_innovation_number += 1

            # add a connection
            if random.random() < self.node_connection_mutation_constants[1]:
                individual.add_connection(
                    self.global_connection_innovation_number
                )

                self.global_connection_innovation_number += 1

        return offspring

    def plot_time_live(self):
        """
        Plot the time history
        """

        y = self.generation_time_hist
        x = list(range(len(y)))

        plt.plot(x, y)
        plt.show(block=False)
        plt.pause(0.001)

    def plot_best_fitness_live(self):
        """
        Plot the fitness of the best history
        """

        y = self.best_fitness_hist
        x = list(range(len(y)))

        plt.plot(x, y)
        plt.show(block=False)
        plt.pause(0.001)

    def plot_fitness_live(self):
        """
        Plot the fitness history
        """

        y = self.fitness_hist
        x = list(range(len(y)))

        plt.plot(x, y)
        plt.show(block=False)
        plt.pause(0.001)

    def save_population(self, file_name=None):
        """
        Saves the current population to a json file
        """

        if file_name == None:
            file_name = (
                f"NEAT-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
            )

        if file_name.split(".")[-1] != "json":
            file_name = (
                f"NEAT-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
            )
            print(
                f"[ERROR] Specified file is not a json file. Saving to '{file_name}' instead."
            )

        if os.path.exists(file_name):
            inp = input(
                f"[WARNING] The file {file_name} already exists. Do you want to proceed? [y/n] "
            ).lower()
            while True:
                if inp == "y":
                    print(f"[INFO] Saving to {file_name}...")
                    break
                elif inp == "n":
                    print("[INFO] Saving aborted")
                    return
                else:
                    inp = input(
                        f"Invalid answer. Do you want to proceed? [y/n] "
                    ).lower()

        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        population_data = {
            "time": time,
            "population": []
        }

        for individual in self.population:
            individual_data = individual.save_network_raw_data()
            population_data["population"].append(individual_data)

        with open(f"NEAT_saves/{file_name}", "w") as f:
            f.write(json.dumps(population_data, indent=4))

        print(f"[INFO] Saved population to {file_name}.")

    @staticmethod
    def load_population(file_name=None):
        """
        This method loads the current network from a file.
        """

        if not file_name:
            file_name = sorted(list(os.listdir('NEAT_saves')))[-1]

        elif not os.path.exists(file_name):
            print("[ERROR] The specified file does not exist")

        with open(f"NEAT_saves/{file_name}", "r") as f:
            data = json.loads(f.read())

        population = []
        for individual_data in data["population"]:
            population.append(
                Genome.load_network_from_raw_data(individual_data)
            )

        print(f"[INFO] Loaded Population from '{file_name}'")

        return population

    def make_population_empty(self, activation_functions_hidden=[], activation_functions_output=[]):
        """
        Makes a generation of the given population size with the given nn_base_dimensions
        The nets in the population are connection free, so they only contain nodes
        """

        self.population = []
        self.global_connection_innovation_number = 0

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
    import sys
    sys.setrecursionlimit(2**15)

    N = NEAT(PongEnvNEAT, 50, (1, 1, 0.4), (0.8, 0.9),
             (0.001, 0.001), 0.1, 0.5, 10000, 0.75)

    N.make_population_connected()
    #N.population = NEAT.load_population()

    N.iterate(20, print_frequency=1)

    N.save_population()

    N.simulate_population(10000)

    """
    for individual in N.population:
        print(individual.fitness)
    """

    for individual in random.sample(N.population, k=5):
        individual.draw()

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


def main2():
    return


if __name__ == "__main__":
    main2()
