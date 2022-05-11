# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

from typing import List


class NodeGene:
	def __init__(self, node_id, node_type, activation_function):
		self.node_id = node_id
		self.node_type = node_type
		
		self.activation_function = activation_function


class ConnectionGene:
	def __init__(self, 
		out_node_id, 
		in_node_id, 
		weight, 
		innovation_number,
		enabled = True
		):

		self.out_node_id = out_node_id
		self.in_node_id = in_node_id
		self.weight = weight
		self.innovation_number = innovation_number
		self.enabled = enabled


class Genome:
	def __init__(self, nodes, connections):
		self.nodes = sorted(nodes, key = lambda x: x.node_id)
		self.connections = connections


	def add_connection(self) -> None:
		'''
		Mutation method that adds a connections between two nodes that were not connected before
		'''
		pass


	def add_node(self) -> None:
		'''
		Mutation method that splits a connection between two nodes and inserts a new node in the
		middle
		'''
		pass


	def calculate_node(self, node):
		'''
		Function to recursively calculate the 
		'''
	
		if node.node_id in self.table:
			return self.table[node.node_id]

		inputs = []

		# if the node value has not been calculated yet, calculate it with all the inputs going in
		# to the node
		for connection in self.connections:
			print(connection.__dict__)
			print(node.node_id)
			print(connection.out_node_id == node.node_id)
			if connection.in_node_id == node.node_id and connection.enabled:
				inputs.append(
					self.calculate_node(
						self.nodes[connection.out_node_id]
					) * connection.weight)

		print(inputs)

		# store the value in the table to 
		activation = node.activation_function(sum(inputs))
		self.table[node.node_id] = activation
		return activation

	def feed_forward(self, input_vector):
		'''
		Feeds the given input through the network induced by the genotype
		'''


		# set up a table to store the activations of the different nodes
		self.table = {}

		for index, input_activation in enumerate(list(input_vector)):
			self.table[index] = input_activation

		for node in self.nodes:
			if node.node_type == 'output':
				self.calculate_node(node)

		return self.table




def main():
	G = Genome(
		[NodeGene(0, 'input', lambda x: x), NodeGene(1, 'input', lambda x: x), NodeGene(2, 'output', lambda x: x)],
		[ConnectionGene(0, 2, -0.5, 0), ConnectionGene(1, 2, 0.5, 1)]
	)

	print([node.__dict__ for node in G.nodes])
	print([connection.__dict__ for connection in G.connections])
	
	print(G.feed_forward([1, 2]))
	

if __name__ == '__main__': main()
