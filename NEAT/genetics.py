# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

import random
import json
import datetime
import os
import networkx as nx
import matplotlib.pyplot as plt

from activation_functions import *


class NodeGene:
	def __init__(self, node_id, node_type, activation_function):
		self.id = node_id

		# specifies the type of the function, either 'input', 'hidden' or 'output'
		self.type = node_type
		
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
		self.nodes = {node.id : node for node in nodes}

		self.connections = connections


	def save_network(self, file_name = None):
		'''
		Method to save a genome to a .json file
		'''

		if file_name == None:
			file_name = f'NEAT-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'

		if file_name.split('.')[-1] != 'json':
			file_name = f'NEAT-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
			print(f'[ERROR] Specified file is not a json file. Saving to \'{file_name}\' instead.')

		if os.path.exists(file_name):
			inp = input(f'[WARNING] The file {file_name} already exists. Do you want to proceed? [y/n] ').lower()
			while True:
				if inp == 'y':
					print(f'[INFO] Saving to {file_name}...')
					break
				elif inp == 'n':
					print('[INFO] Saving aborted')
					return
				else:
					inp = input(f'Invalid answer. Do you want to proceed? [y/n] ').lower()

		data = {
			'connections': [connection.__dict__ for connection in self.connections], 
			'nodes': {node_id: node.__dict__ for node_id, node in self.nodes.items()}
		}
		
		json_data = json.dumps(data, indent=4)	

		with open(file_name, 'w') as f:
			f.write(json_data)

		print(f'[INFO] Saved data to \'{file_name}\'')


	def load_network(self, file_name) -> None:
		'''
		This method loads the current network from a file.
		'''

		if not os.path.exists(file_name):
			print('[ERROR] The specified file does not exist')

		with open(file_name, 'r') as f:
			data = json.loads(f.read())

		connections = [ConnectionGene(**params) for params in data['connections']]

		nodes = {node_id: NodeGene(**params) for node_id, params in data['nodes'].items()}

		self.__init__(list(nodes.values()), connections)

		print(f'[INFO] loaded Neural Network from \'{file_name}\'')


	def add_connection(self, innovation_number):
		'''
		Mutation method that adds a connections between two nodes that were not connected before
		'''
		
		done = False
		while not done:
			in_node, out_node = random.sample(list(self.nodes.values()), k = 2)

			# the new connection cannot go between input layer node or output layer nodes
			# in addition, they must not lead to an input or come from an output node
			if not ((in_node.type == 'input' and out_node.type == 'input') or 
				(in_node.type == 'output' and out_node.type == 'output') or
				(out_node.type == 'output') or (in_node.type == 'input')):
				done = True

			# check if connection does not already exist
			for connection in self.connections:
				if ((connection.in_node_id == in_node.id and 
					connection.out_node_id == out_node.id) or
					(connection.out_node_id == in_node.id and 
					connection.in_node_id == out_node.id)):
					done = False

		self.connections.append(
			ConnectionGene(out_node.id, in_node.id, random.random(), innovation_number)
		)
		

	def add_node(self, node_id, innovation_number, activation_function = 'linear'):
		'''
		Mutation method that splits a connection between two nodes and inserts a new node in the
		middle
		'''

		new_node = NodeGene(node_id, 'hidden', activation_function)
		self.nodes[node_id] = new_node

		connection = random.choice(self.connections)
		connection.enabled = False

		out_node_id = connection.out_node_id
		connection_to_new_node = ConnectionGene(
			out_node_id, 
			node_id, 
			1, 
			innovation_number
		)
		self.connections.append(connection_to_new_node)

		in_node_id = connection.in_node_id
		connection_from_new_node = ConnectionGene(
			node_id, 
			in_node_id, 
			connection.weight, 
			innovation_number + 1
		)
		self.connections.append(connection_from_new_node)
		
		


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
			if node.type == 'output':
				self.calculate_node(node_id)

		return self.table


	def draw(self, weight = False):
		Graph = nx.Graph()

		c = (i for i in range(0, len(self.nodes)))
		for node in self.nodes.values():
			if node.type == 'input':
				Graph.add_node(node.id, pos=(0, len(self.nodes)//2 + node.id))
			elif node.type == 'output':
				Graph.add_node(node.id, pos=(10, node.id))
			else:
				Graph.add_node(node.id, pos=(5, next(c)))

		for connection in self.connections:
			Graph.add_edge(
				connection.in_node_id, 
				connection.out_node_id,
				weight=connection.weight,
				color='tab:blue' if connection.enabled else 'tab:red'
			)

		pos = nx.get_node_attributes(Graph, 'pos')
		edge_color = nx.get_edge_attributes(Graph, 'color').values()
		nx.draw_networkx(Graph, pos=pos, with_labels=True, edge_color=edge_color)
		
		if weight:
			labels = nx.get_edge_attributes(Graph, 'weight') 
		else:
			labels = {e: '' for e in Graph.edges}
		nx.draw_networkx_edge_labels(Graph, pos=pos, edge_labels=labels)

		plt.show()



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

	'''
	G.add_connection(10)
	for connection in G.connections:
		print(connection.__dict__)

	G.load_network('test.json')	
	'''

	G.add_node(10, 20)

	for connection in G.connections:
		print(connection.__dict__)

	G.draw()

if __name__ == '__main__': main()
