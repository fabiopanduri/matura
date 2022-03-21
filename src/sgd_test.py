from neural_network import *

def function(x, a, b, c) -> float:
    return a * x ** 2 + b * x + c

def main():
    NN = NeuralNetwork([4, 4, 1], 0.9, activation_functions = [np.vectorize(sigmoid) for _ in range(2)] + [np.vectorize(ReLU)])
    NN.initialize_network()

    a = 10
    b = -3
    c = 14

    train = []
    for i in range(1000):
        x = np.random.rand()
        train.append((np.array([a, b, c, x]), np.array([function(x, a, b, c)])))
        
    print(NN.feed_forward(train[0][0]), train[0])

    print(NN.weights)
    NN.stochastic_gradient_descent(train)
    print(NN.weights)

    test = []
    for i in range(0):
        x = np.random.rand()
        test.append((np.array([a, b, c, x]), np.array([function(x, a, b, c)])))

    for t in test:
        print(NN.feed_forward(t[0]), t)

if __name__ == '__main__': main()
