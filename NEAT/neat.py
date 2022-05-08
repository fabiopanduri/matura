# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

from typing import List


class NodeGene:
	def __init__(self, node_id: int, node_type: str):
		self.node_id = node_id
		self.node_type = node_type


class ConnectionGene:
	def __init__(self, 
		out_node: int, 
		in_node: int, 
		weight: float, 
		innovation_number: int,
		enabled: bool = True
		):

		self.out_node = out_node
		self.in_node = in_node 
		self.weight = weight
		self.innovation_number = innovation_number
		self.enabled = enabled


class Genome:
	def __init__(self, nodes: List[NodeGene], connections: List[ConnectionGene]):
		self.nodes = nodes
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

