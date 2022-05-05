import numpy as np

from typing import List, Callable


def sigmoid(x):
	'''
	Sigmoid activation function.
	'''

	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	'''
	Derivative of the sigmoid activation function.  
	'''

	return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
	'''
	ReLU activation function.
	'''

	return max(0, x)

def ReLU_derivative(x):
	'''
	Derivative of the ReLU activation function.
	'''

	return 1 if x > 0 else 0

def mean_squared_error_derivative(activation, y):
	'''
	Derivative of the cost function (mean squared error) with respect to the activation.
	Here the derivative of Mean squared error.
	'''

	return activation - y


class NeuralNetwork:
	'''
	This class includes all functionalities for a neural network.
	'''

	def __init__(self, dimensions: List[int], eta: float, weights: List['numpy_array'] = [], biases: List['numpy_array'] = [], activation_functions: List[Callable] = [], activation_functions_derivatives: List[Callable] = []) -> None:

		# list of dimensions of the neural network
		self.dimensions = dimensions

		# number of layers the neural network has
		self.layers: int = len(self.dimensions)

		# list of all the weight matrices for the neural net
		# entry at l is w^{l}
		self.weights = weights

		# list of all the bias vectors for the neural net
		# entry at l is b^{l}
		self.biases = biases

		# list containing all the activation functions for each layer
		# entry at l is \sigma^{l}
		self.activation_functions = activation_functions

		# list containing all the derivaties of the activation functions for each layer
		# entry at l is \sigma^{l}\prime
		self.activation_functions_derivatives = activation_functions_derivatives
		
		# learning rate eta for gradient descent
		self.eta = eta


	def initialize_network(self) -> None:
		'''
		This method randomly initializes the weights and the biases.
		'''

		# add empty array to self.weights so that the index corresponds to the layer
		self.weights = [np.array([])]

		# initialize the weights matrices w^{l} with dimensions dim(l) x dim(l - 1)
		for l, dim in enumerate(self.dimensions[1:], 1):
			self.weights.append(np.random.uniform(0, 1, (dim, self.dimensions[l - 1])))

		# add empty array to self.weights so that the index corresponds to the layer
		self.biases = [np.array([])]

		# initialize the bias vectors b^{l} with dimensions dim(l)
		for dim_l in self.dimensions[1:]:
			self.biases.append(np.random.uniform(0, 1, dim_l))


	def feed_forward(self, input_vector: 'numpy_array') -> 'numpy_array':
		'''
		This method feeds the input_vector through the network.
		'''

		activation = input_vector
		# for each layer compute the activation
		for l in range(1, self.layers):
			z_l = np.dot(self.weights[l], activation) + self.biases[l]
			activation = self.activation_functions[l](z_l)

		# return the activation for the output layer
		return activation


	def stochastic_gradient_descent(self, training_batch) -> None:
		'''
		This method implements the stochastic gradient descent algorithm.
		'''

		# initialize lists to store the sum of errors for each weight and bias
		bias_delta_sum = [np.zeros(self.dimensions[l]) for l in range(self.layers)]

		weight_delta_sum = [0] + [np.zeros((self.dimensions[l], self.dimensions[l - 1])) for l in range(1, self.layers)]

		# iterate over all training examples
		for training_example in training_batch:
			#print(f'{training_example=}')
			activation = training_example[0]

			# set up list to store all the z vectors for later use
			# initialized with z vector at 0 equal to 0 for indexing purposes
			z_list = [0]

			# set up a list to store all the activation vectors for each layer for later
			activation_list = []
			activation_list.append(activation)

			# calculate the activation and z vector for all the layers
			for l in range(1, self.layers):
				#print(f'\n{l=}\n{z_list=}\n{activation_list=}\n')
				# compute the z vector and store it in the z_list 
				z = np.dot(self.weights[l], activation) + self.biases[l]
				z_list.append(z)

				# compute the activation of the l-th layer and add it to the activation list
				activation = self.activation_functions[l](z)
				activation_list.append(activation)

			# calculate the error of the last layer
			delta_L = mean_squared_error_derivative(activation, training_example[1]) * self.activation_functions_derivatives[-1](z)

			# calculate the error of each other layer
			delta = [np.zeros(self.dimensions[l]) for l in range(self.layers)]
			for l in range(self.layers - 1, 0, -1):

				if l == self.layers - 1:
					delta[-1] = delta_L
				else:
					delta[l] = np.dot(self.weights[l + 1].T, delta[l + 1]) * self.activation_functions_derivatives[l](z_list[l])

				'''
				print(activation_list[l - 1][..., None].T)
				print(delta[l][..., None])
				print([w.ndim for w in weight_delta_sum[l]])
				print(weight_delta_sum[l])
				print(self.weights[l])
				print(np.dot(delta[l][..., None], activation_list[l - 1][..., None].T))
				print(np.dot(delta[l], activation_list[l - 1].T))
				'''

				# update the sum of errors for weights and biases for each layer
				weight_delta_sum[l] = weight_delta_sum[l] + np.dot(delta[l][..., None], activation_list[l - 1][..., None].T)

				bias_delta_sum[l]  = bias_delta_sum[l] + delta[l]


		# update the weights and biases according to the calculated errors
		for l in range(1, self.layers):
			self.weights[l] = self.weights[l] - weight_delta_sum[l] * self.eta / len(training_batch)
			self.biases[l] = self.biases[l] - bias_delta_sum[l] * self.eta / len(training_batch)





def main():
	NN = NeuralNetwork([2, 2, 1], 0.9, weights = [np.array([]), np.array([[0, 1], [1, 0]]), np.array([2, 3])],
		biases = [np.array([]), np.array([0]), np.array([0])],
		activation_functions = [0, np.vectorize(lambda x: x), np.vectorize(lambda x: x)])
	#NN.initialize_network()
	print(NN.feed_forward(np.array([2, 1])))

if __name__ == '__main__': main()
