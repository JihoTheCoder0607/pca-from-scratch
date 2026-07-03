import numpy as np

rng = np.random.default_rng(seed=42)

def relu(x):
    return max(0, x)

def leaky_relu(x, a=0.01):
    if x >= 0:
        return x
    else:
        return a * x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

class Layer:
    def __init__(self, n_in, n_out, activation):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if activation == "relu":
            # He initialization
            self.weights = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        elif activation in ("sigmoid", "tanh"):
            # Xavier initialization
            self.weights = rng.standard_normal((n_in, n_out)) * np.sqrt(1.0 / n_in)
        else:
            self.weights = rng.standard_normal((n_in, n_out)) * 0.01

        self.biases = np.zeros((1, n_out))

class NN:
    def __init__(self, input_shape, layers=None):
        self.input_shape = input_shape
        self.layers = layers if layers is not None else []
    def add(self, layer):
        self.layers.append(layer)