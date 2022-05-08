# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

from neural_network import *
import matplotlib.pyplot as plt
import time
import sys

def f(x) -> float:
	return (np.cos(x) + 1) / 2

def main():
	NN = NeuralNetwork([1, 4, 1], 0.1,
			activation_functions = ['ReLU', 'ReLU', 'sigmoid'],
			)
	NN.initialize_network()

	START = 0
	STOP = np.pi 

	try:
		epochs = int(sys.argv[1])
	except IndexError:
		epochs = 100

	try:
		batch_size = int(sys.argv[2])
	except IndexError:
		batch_size = 1000


	train = []
	for _ in range(batch_size):
		#a, b, c = np.random.uniform(), np.random.rand(), np.random.rand()
		x = np.random.uniform(START, STOP)
		train.append((np.array([x]), np.array([f(x)])))

	#plt.figure()
	#plt.plot([i[0] for i in train], [i[1] for i in train], ',')

	error = []
	print('')
	for i in range(epochs):
		if i % 100 == 0: print(f'{i}/{epochs}', end='\r')

		NN.stochastic_gradient_descent(train)
		
	#print(NN.feed_forward(train[0][0]), train[0])

	#print(NN.weights)
	#print(NN.weights)
		#print(NN.weights)

		test = []
		for _ in range(10):
			x = np.random.uniform(START, STOP)
			test.append((np.array([x]), np.array([f(x)])))

		sum = 0
		for t in test:
			prediction = NN.feed_forward(t[0])
			 #print(prediction)
			sum += 0.5 * (prediction - t[1])**2

		#print(sum)

		#print(sum / len(test))
		#print(i)
		error.append(sum / len(test))

	print(f'{NN.weights=}')
	print(f'{NN.biases=}')

	predict = []
	for x in np.arange(START, STOP, 0.01):
		print(x, NN.feed_forward(np.array([x])))
		predict.append(NN.feed_forward(np.array([x])))

	actual = [f(x) for x in np.arange(START, STOP, 0.01)]

	#print(predict)

	plt.figure()
	plt.plot(np.arange(START, STOP, 0.01), predict)
	plt.plot(np.arange(START, STOP, 0.01), actual)
	plt.figure()
	plt.plot(list(range(0, len(error))), error)
	plt.show()

if __name__ == '__main__': main()
