# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
"""
Genetics for NEAT
"""
import datetime
import json
import os
import random

import matplotlib.pyplot as plt
import networkx as nx

import etc.activation_functions as af


class NodeGene:
    def __init__(self, node_id, node_type, activation_function):
        self.id = node_id

        # specifies the type of the function, either 'input', 'hidden' or 'output'
        self.type = node_type

        # get the activation function by name
        self.activation_function = af.activation_functions[activation_function]

        self.activation_function_name = activation_function

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.id == other.id
        return False

    @property
    def __dict__(self):
        d = {
            "node_id": self.id,
            "node_type": self.type,
            "activation_function": self.activation_function_name
        }

        return d


class ConnectionGene:
    def __init__(
        self, out_node_id, in_node_id, weight, innovation_number, enabled=True
    ):

        # specifies the tail of the connection arc
        self.out_node_id = out_node_id

        # specifies the head of the connection arc
        self.in_node_id = in_node_id

        self.weight = weight

        self.innovation_number = innovation_number

        self.enabled = enabled


class Genome:
    def __init__(self, nodes, connections):
        self.nodes = {node.id: node for node in nodes}

        self.output_nodes = {}
        for node in nodes:
            if node.type == "output":
                self.output_nodes[node.id] = node

        self.connections = connections

        self.fitness = 0

        self.total_rec_depth = 0

    def connection_dict(self):
        return {c.innovation_number: c for c in self.connections}

    @classmethod
    def make_empty_genome(
        cls,
        n_inputs,
        n_hidden,
        n_outputs,
        activation_functions_hidden=[],
        activation_functions_output=[],
    ):

        nodes = []
        node_id = 0

        for _ in range(n_inputs):
            nodes.append(NodeGene(node_id, "input", "linear"))
            node_id += 1

        for i in range(n_hidden):
            try:
                act_func = activation_functions_hidden[i]
            except IndexError:
                act_func = "sigmoid"

            nodes.append(NodeGene(node_id, "hidden", act_func))
            node_id += 1

        for i in range(n_outputs):
            try:
                act_func = activation_functions_output[i]
            except IndexError:
                act_func = "sigmoid"

            nodes.append(NodeGene(node_id, "output", act_func))
            node_id += 1

        return cls(nodes, [])

    @classmethod
    def make_connected_genome(cls, n_inputs, n_outputs, activation_functions_output=[]):
        """
        Method to crate a network where all input nodes have a connection to all output nodes
        """

        nodes = []
        node_id = 0

        for _ in range(n_inputs):
            nodes.append(NodeGene(node_id, "input", "linear"))
            node_id += 1

        for i in range(n_outputs):
            try:
                act_func = activation_functions_output[i]
            except IndexError:
                act_func = "sigmoid"

            nodes.append(NodeGene(node_id, "output", act_func))
            node_id += 1

        connections = []
        innov = 0
        for out_node in nodes:
            for in_node in nodes:
                if out_node != in_node and out_node.type == 'input' and in_node.type == 'output':
                    connections.append(
                        ConnectionGene(out_node.id, in_node.id,
                                       random.uniform(-2, 2), innov)
                    )
                    innov += 1

        return cls(nodes, connections)

    @classmethod
    def crossover(cls, parent1, parent2, connection_disable_probability):
        # get all nodes from both parents
        nodes = parent1.nodes.copy()
        for node_id, node in parent2.nodes.items():
            if node_id not in nodes:
                nodes[node_id] = node

        connections = []
        p1_connections = parent1.connection_dict()
        p2_connections = parent2.connection_dict()

        # make sure none of the connections are empty
        if len(p1_connections) == 0 or len(p2_connections) == 0:
            connections = parent1.connections + parent2.connections
            return cls(nodes.values(), connections)

        m = max(max(p1_connections.keys()), max(p2_connections.keys()))
        for i in range(m + 1):
            if i in p1_connections and i in p2_connections:
                if parent1.fitness > parent2.fitness:
                    new_connection = p1_connections[i]
                elif parent1.fitness < parent2.fitness:
                    new_connection = p2_connections[i]
                else:
                    new_connection = random.choice(
                        [p1_connections[i], p2_connections[i]])

                # chance that the connection is disabled if either parent had it disabled
                if (not p1_connections[i].enabled or not p2_connections[i].enabled) and random.random() < connection_disable_probability:
                    new_connection.enabled = False
                else:
                    new_connection.enabled = True
                connections.append(new_connection)

            elif i in p1_connections and i not in p2_connections:
                connections.append(p1_connections[i])
            elif i not in p1_connections and i in p2_connections:
                connections.append(p2_connections[i])

        return cls(nodes.values(), connections)

    def delta(self, other, speciation_constants=(1, 1, 1)):
        """
        This method gives the compatibility distance delta of the instance and another instance
        """

        if not isinstance(other, type(self)):
            raise ValueError(f"Type must be {type(self)} not {type(self)}.")

        self_connections = self.connection_dict()
        other_connections = other.connection_dict()
        self_biggest = max(self_connections.keys()) if len(
            self_connections) > 0 else 0
        other_biggest = max(other_connections.keys()) if len(
            other_connections) > 0 else 0

        N = max([len(self_connections), len(other_connections), 1])
        E, D = 0, 0
        weight_difference_sum = 0
        weight_difference_count = 0

        m = max(self_biggest, other_biggest)
        for i in range(m + 1):
            if i in self_connections and i in other_connections:
                weight_difference_sum += abs(
                    self_connections[i].weight - other_connections[i].weight
                )
                weight_difference_count += 1
            elif i in self_connections and i not in other_connections:
                if i > other_biggest:
                    E += 1
                else:
                    D += 1
            elif i not in self_connections and i in other_connections:
                if i > self_biggest:
                    E += 1
                else:
                    D += 1
        if weight_difference_count != 0:
            W_bar = weight_difference_sum / weight_difference_count
        else:
            W_bar = 0

        c_1, c_2, c_3 = speciation_constants

        return (c_1 * E) / N + (c_2 * D) / N + c_3 * W_bar

    def adjust_fitness(self, species):
        """
        This method updates the fitness of the genome according to the number of individuals in the
        species
        """

        self.fitness /= len(species)

    def save_network_raw_data(self):
        """
        Method to generate the dictionary that can be parsed to the NEAT class to save a population
        """

        data = {
            "connections": [connection.__dict__ for connection in self.connections],
            "nodes": {node_id: node.__dict__ for node_id, node in self.nodes.items()},
        }

        return data

    def save_network(self, file_name=None):
        """
        Method to save a genome to a .json file
        """

        if file_name == None:
            file_name = (
                f"NEAT-NN-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
            )

        if file_name.split(".")[-1] != "json":
            file_name = (
                f"NEAT-NN-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
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

        data = {
            "connections": [connection.__dict__ for connection in self.connections],
            "nodes": {node_id: node.__dict__ for node_id, node in self.nodes.items()},
        }

        json_data = json.dumps(data, indent=4)

        with open(file_name, "w") as f:
            f.write(json_data)

        print(f"[INFO] Saved data to '{file_name}'")

    @classmethod
    def load_network_from_raw_data(cls, data) -> None:
        """
        This method loads a network from a parsed data coming from the NEAT class
        """

        connections = [ConnectionGene(**params)
                       for params in data["connections"]]

        nodes = {
            node_id: NodeGene(**params) for node_id, params in data["nodes"].items()
        }

        new_instance = cls(list(nodes.values()), connections)

        return new_instance

    @classmethod
    def load_network(cls, file_name) -> None:
        """
        This method loads network from a file
        """

        if not os.path.exists(file_name):
            print("[ERROR] The specified file does not exist")

        with open(file_name, "r") as f:
            data = json.loads(f.read())

        connections = [ConnectionGene(**params)
                       for params in data["connections"]]

        nodes = {
            node_id: NodeGene(**params) for node_id, params in data["nodes"].items()
        }

        new_instance = cls(list(nodes.values()), connections)

        print(f"[INFO] Loaded Neural Network from '{file_name}'")

        return new_instance

    def has_cycle(self, pot_out_node, pot_in_node):
        """
        Check if a new connection would introduce a cycle
        """

        for connection in self.connections:
            if connection.out_node_id == pot_in_node:
                if connection.in_node_id == pot_out_node:
                    return True
                elif self.has_cycle(pot_out_node, connection.in_node_id):
                    return True

        return False

    def add_connection(self, innovation_number, thresh=100):
        """
        Mutation method that adds a connections between two nodes that were not connected before
        """

        done = False
        i = 0
        while not done:
            in_node, out_node = random.sample(list(self.nodes.values()), k=2)

            # the new connection cannot go between input layer node or output layer nodes
            # in addition, they must not lead to an input or come from an output node
            if not (
                (in_node.type == "input" and out_node.type == "input")
                or (in_node.type == "output" and out_node.type == "output")
                or (out_node.type == "output")
                or (in_node.type == "input")
            ):
                done = True

            # check if connection does not already exist
            for connection in self.connections:
                if (
                    connection.in_node_id == in_node.id
                    and connection.out_node_id == out_node.id
                ) or (
                    connection.out_node_id == in_node.id
                    and connection.in_node_id == out_node.id
                ):
                    done = False
                    continue

            # make sure the new graph would not create a cycle to avoid infinite loops
            if self.has_cycle(out_node.id, in_node.id):
                done = False

            i += 1
            if i > thresh:
                return

        self.connections.append(
            ConnectionGene(out_node.id, in_node.id,
                           random.random(), innovation_number)
        )

    def add_node(self, innovation_number_nodes, innovation_number_connections, activation_function="sigmoid"):
        """
        Mutation method that splits a connection between two nodes and inserts a new node in the
        middle
        """

        if len(self.connections) == 0:
            return

        node_id = innovation_number_nodes
        new_node = NodeGene(node_id, "hidden", activation_function)
        self.nodes[node_id] = new_node

        connection = random.choice(self.connections)
        connection.enabled = False

        out_node_id = connection.out_node_id
        connection_to_new_node = ConnectionGene(
            out_node_id, node_id, 1, innovation_number_connections
        )
        self.connections.append(connection_to_new_node)

        in_node_id = connection.in_node_id
        connection_from_new_node = ConnectionGene(
            node_id, in_node_id, connection.weight, innovation_number_connections + 1
        )
        self.connections.append(connection_from_new_node)

    def mutate_weights(self, weight_mutation_constants=(0.8, 0.9)):
        """
        With a given chance mutate a weight either by multiplying it with a number or by getting a
        completely new value
        First element in weight_mutation_constants is the chance a weight is mutated
        The second one is the chance it is multiplied if it is mutated
        """

        mut_chance, mult_chance = weight_mutation_constants

        for connection in self.connections:
            if random.random() < mut_chance:
                if random.random() < mult_chance:
                    connection.weight *= random.uniform(-2, 2)
                else:
                    connection.weight = random.uniform(-2, 2)

    def calculate_node(self, node_id):
        """
        Method to recursively calculate the activation of nodes
        """

        if node_id in self.table:
            return self.table[node_id]

        inputs = []

        # if the node value has not been calculated yet, calculate it with all the inputs going in
        # to the node
        for connection in self.connections:
            if connection.in_node_id == node_id and connection.enabled:
                self.total_rec_depth += 1
                if self.total_rec_depth > 2**11:
                    # print("alarm")
                    pass
                inputs.append(
                    self.calculate_node(
                        connection.out_node_id
                    ) * connection.weight
                )

        activation = self.nodes[node_id].activation_function(sum(inputs))
        self.table[node_id] = activation
        return activation

    def feed_forward(self, input_vector):
        """
        Feeds the given input through the network induced by the genotype
        """

        # set up a table to store the activations of the different nodes
        self.table = {}

        # store the input layer activation in the table
        for index, input_activation in enumerate(list(input_vector)):
            self.table[index] = input_activation

        # iterate over all output nodes and calculate their activation recursively
        for node_id in self.output_nodes.keys():
            self.calculate_node(node_id)
            self.total_rec_depth = 0

        return [self.table[i] for i in sorted(list(self.output_nodes.keys()))]

    def draw(self, weight=False):
        Graph = nx.Graph()

        c = (i for i in range(0, len(self.nodes)))
        for node in self.nodes.values():
            if node.type == "input":
                Graph.add_node(node.id, pos=(
                    0, 4 * node.id))
            elif node.type == "output":
                Graph.add_node(node.id, pos=(10, 2 * node.id))
            else:
                Graph.add_node(node.id, pos=(random.randint(3, 7), next(c)))

        for connection in self.connections:
            Graph.add_edge(
                connection.out_node_id,
                connection.in_node_id,
                weight=connection.weight,
                color="black" if connection.enabled else "tab:red",
            )

        pos = nx.get_node_attributes(Graph, "pos")
        edge_color = nx.get_edge_attributes(Graph, "color").values()
        nx.draw_networkx(Graph, pos=pos, with_labels=False, edge_color=edge_color, arrowstyle="->",
                         arrowsize=10, arrows=True, width=1.2, node_size=400)

        if weight:
            labels = nx.get_edge_attributes(Graph, "weight")
        else:
            labels = {e: "" for e in Graph.edges}
        nx.draw_networkx_edge_labels(Graph, pos=pos, edge_labels=labels)

        plt.style.use("ggplot")
        plt.show()
