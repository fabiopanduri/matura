from neural_network import *
import matplotlib.pyplot as plt

def function(x, a, b, c) -> float:
    return a * x ** 2 + b * x + c

def function_2(a, b, c) -> float: 
    return a + b + c

def main():
    NN = NeuralNetwork([3, 1], 0.1, 
            activation_functions = [np.vectorize(sigmoid) for _ in range(3)],
            activation_functions_derivatives = [np.vectorize(sigmoid_derivative) for _ in range(3)]
            )
    NN.initialize_network()

    a = 10
    b = -3
    c = 14
    
    error = []
    for i in range(200):
        train = []
        for _ in range(10000):
            a, b, c = np.random.rand(), np.random.rand(), np.random.rand()
            train.append((np.array([a, b, c]), np.array([function_2(a, b, c)])))

        NN.stochastic_gradient_descent(train)
        
    #print(NN.feed_forward(train[0][0]), train[0])

    #print(NN.weights)
    #print(NN.weights)
        #print(NN.weights)

        test = []
        for _ in range(100):
            a, b, c = np.random.rand(), np.random.rand(), np.random.rand()
            test.append((np.array([a, b, c]), np.array([function_2(a, b, c)])))

        sum = 0
        for t in test:
            prediction = NN.feed_forward(t[0])
            sum += abs(prediction - t[1])

        #print(sum / len(test))
        print(i)
        error.append(sum / len(test))

    plt.plot(list(range(0, len(error))), error)
    plt.show()

if __name__ == '__main__': main()
