import numpy as np

rng = np.random.default_rng(seed=42)

# activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, a=0.01):
    return np.where(x >= 0, x, a * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# loss functions
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def binary_cross_entropy(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def categorical_cross_entropy(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    return -np.mean(np.sum(y * np.log(y_pred), axis=-1))

class Layer:
    def __init__(self, n_in, n_out, activation):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if activation == relu:
            # He initialization
            self.weights = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        elif activation in (sigmoid, tanh):
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
    def forward(self, X):
        for layer in self.layers:
            X_forward = X @ layer.weights + layer.biases
            for row in range(X_forward.shape[0]):
                X_forward[row] = layer.activation(X_forward[row])
        return X_forward