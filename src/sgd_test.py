from neural_network import *
import matplotlib.pyplot as plt
import time

def function(x, a, b, c) -> float:
    return a * x ** 2 + b * x + c

def function_2(a, b, c) -> float: 
    return a + b + c

def main():
    NN = NeuralNetwork([1, 8, 8, 8, 1], 0.1,
            activation_functions = [np.vectorize(sigmoid) for _ in range(5)],
            activation_functions_derivatives = [np.vectorize(sigmoid_derivative) for _ in range(5)]
            )
    NN.initialize_network()

    error = []
    for i in range(1000):
        train = []
        for _ in range(10000):
            #a, b, c = np.random.uniform(), np.random.rand(), np.random.rand()
            x = np.random.uniform(0, 2*np.pi)
            train.append((np.array([x]), np.array([np.cos(x)])))

        time.sleep(0.5)

        NN.stochastic_gradient_descent(train)
        
    #print(NN.feed_forward(train[0][0]), train[0])

    #print(NN.weights)
    #print(NN.weights)
        #print(NN.weights)

        test = []
        for _ in range(100):
            #a, b, c = np.random.uniform(), np.random.rand(), np.random.rand()
            x = np.random.uniform(0, 2*np.pi)
            test.append((np.array([x]), np.array([np.cos(x)])))

        sum = 0
        for t in test:
            prediction = NN.feed_forward(t[0])
            sum += abs(prediction - t[1])

        #print(sum / len(test))
        print(i)
        error.append(sum / len(test))

    print(f'{NN.weights=}')
    print(f'{NN.biases=}')

    predict = []
    for x in np.arange(0, 2*np.pi, 0.001):
        predict.append(NN.feed_forward(np.array([x])))

    actual = [np.cos(x) for x in np.arange(0, 2*np.pi, 0.001)]

    #print(predict)

    plt.figure()
    plt.plot(np.arange(0, 2*np.pi, 0.001), predict)
    plt.plot(np.arange(0, 2*np.pi, 0.001), actual)
    plt.figure()
    plt.plot(list(range(0, len(error))), error)
    plt.show()

if __name__ == '__main__': main()
