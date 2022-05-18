# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

import random
import json

from activation_functions import *


class NodeGene:
	def __init__(self, node_id, node_type, activation_function):
		self.node_id = node_id

		# specifies the type of the function, either 'input', 'hidden' or 'output'
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

		# specifies the tail of the connection arc 
		self.out_node_id = out_node_id

		# specifies the head of the connection arc 
		self.in_node_id = in_node_id

		self.weight = weight

		self.innovation_number = innovation_number

		self.enabled = enabled


class Genome:
	def __init__(self, nodes, connections):
		self.nodes = {node.node_id : node for node in nodes}

		self.connections = connections


	def save(self, filename = None):
		'''
		Method to save a genome to a .npz file
		'''
		pass

		data = {
			'connections': [connection.__dict__ for connection in self.connections], 
			'nodes': {node_id: node.__dict__ for node_id, node in self.nodes.items()}
		}
		
		json_data = json.dumps(data, indent=4)	

		print(json_data)


	def add_connection(self, innovation_number):
		'''
		Mutation method that adds a connections between two nodes that were not connected before
		'''
		
		done = False
		while not done:
			in_node, out_node = random.sample(list(self.nodes.values()), k = 2)

			# the new connection cannot go between input layer node or output layer nodes
			# in addition, they must not lead to an input or come from an output node
			if not ((in_node.node_type == 'input' and out_node.node_type == 'input') or 
				(in_node.node_type == 'output' and out_node.node_type == 'output') or
				(out_node.node_type == 'output') or (in_node.node_type == 'input')):
				done = True

			# check if connection does not already exist
			for connection in self.connections:
				if ((connection.in_node_id == in_node.node_id and 
					connection.out_node_id == out_node.node_id) or
					(connection.out_node_id == in_node.node_id and 
					connection.in_node_id == out_node.node_id)):
					done = False

		self.connections.append(
			ConnectionGene(out_node.node_id, in_node.node_id, random.random(), innovation_number)
		)
		

	def add_node(self):
		'''
		Mutation method that splits a connection between two nodes and inserts a new node in the
		middle
		'''
		pass


	def calculate_node(self, node_id):
		'''
		Function to recursively calculate the activation of nodes
		'''
	
		if node_id in self.table:
			return self.table[node_id]

		inputs = []

		# if the node value has not been calculated yet, calculate it with all the inputs going in
		# to the node
		for connection in self.connections:
			if connection.in_node_id == node_id and connection.enabled:
				inputs.append(
					self.calculate_node(
						connection.out_node_id
					) * connection.weight)

		activation = self.nodes[node_id].activation_function(sum(inputs))
		self.table[node_id] = activation
		return activation


	def feed_forward(self, input_vector):
		'''
		Feeds the given input through the network induced by the genotype
		'''

		# set up a table to store the activations of the different nodes
		self.table = {}

		# store the input layer activation in the table
		for index, input_activation in enumerate(list(input_vector)):
			self.table[index] = input_activation
	
		# iterate over all output nodes and calculate their activation recursively
		for node_id, node in self.nodes.items():
			if node.node_type == 'output':
				self.calculate_node(node_id)

		return self.table




def main():
	G = Genome(
		[
			NodeGene(0, 'input', 'linear'), 
			NodeGene(1, 'hidden', 'linear'), 
			NodeGene(2, 'hidden', 'linear'), 
			NodeGene(3, 'hidden', 'linear'),
			NodeGene(4, 'output', 'linear')
		],
		[
			ConnectionGene(0, 1, -0.5, 0), 
			ConnectionGene(0, 2, 0.5, 1), 
			ConnectionGene(1, 4, 0.5, 2),
			ConnectionGene(2, 4, 2, 3)
		]
	)


	'''
	print([node.__dict__ for node in G.nodes.values()])
	print([connection.__dict__ for connection in G.connections])
	
	print(G.feed_forward([1]))
	'''

	G.add_connection(10)
	for connection in G.connections:
		print(connection.__dict__)

	G.save()	

if __name__ == '__main__': main()
