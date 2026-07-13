import numpy as np
from pca import mean

rng = np.random.default_rng(seed=42)

# activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, a=0.01):
    return np.where(x >= 0, x, a * x)

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# derivatives of activation functions
def relu_d(x):
    return np.where(
        x >= 0,
        1,
        0
    )

def leaky_relu_d(x, a=0.01):
    return np.where(
        x >= 0,
        1,
        a
    )

def tanh_d(x):
    return 1-(tanh(x)**2)

def sigmoid_d(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)

def softmax_d(x):
    def softmax_jacobian(s):
        s = s.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

# loss functions
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def binary_cross_entropy(y, y_probs):
    y_pred = np.clip(y_probs, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def categorical_cross_entropy(y, y_probs):
    y_probs = np.clip(y_probs, 1e-15, 1.0)
    return -np.mean(np.sum(y * np.log(y_probs), axis=-1))

# derivatives of loss functions
def mse_d(y, y_pred):
    return (y_pred-y) / y.shape[0]

def mae_d(y, y_pred):
    return np.sign(y_pred-y) / y.shape[0]

def sigmoid_BCE_d(y, y_probs):
    return (y_probs-y) / y.shape[0]

def softmax_CCE_d(y, y_probs):
    return (y_probs-y) / y.shape[0]

# metrics
def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))

def r2(y, y_pred):
    SSres = np.sum((y-y_pred) ** 2)

    y_mean = mean(y)
    SStot = np.sum((y-y_mean) ** 2)

    return 1 - SSres/SStot

def confusion_matrix(y, y_pred):
    classes = np.unique(y)
    num_classes = classes.shape[0]

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y, y_pred):
        conf_matrix[true, pred] += 1
    return conf_matrix

def accuracy(y, y_pred):
    return mean(y == y_pred)

class Layer:
    def __init__(self, n_in, n_out, activation, activation_d):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.activation_d = activation_d

        if self.activation == relu:
            # He initialization
            self.weights = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        elif self.activation in (sigmoid, tanh):
            # Xavier initialization
            self.weights = rng.standard_normal((n_in, n_out)) * np.sqrt(1.0 / n_in)
        else:
            self.weights = rng.standard_normal((n_in, n_out)) * 0.01

        self.biases = np.zeros((1, n_out))
    def forward(self, X):
        self.input = X
        self.Z = X @ self.weights + self.biases
        self.A = self.activation(self.Z)
        return self.A
    def backward(self, delta):
        delta = delta * self.activation_d(self.Z)
        self.dLdW = self.input.T @ delta
        self.dLdb = np.sum(delta, axis=0, keepdims=True)

        return delta @ self.weights.T

class NN:
    def __init__(self, input_shape, layers=None):
        self.input_shape = input_shape
        self.layers = layers if layers is not None else []
    def add(self, layer):
        self.layers.append(layer)
    def compile(self, loss, loss_d, metrics):
        self.loss = loss
        self.loss_d = loss_d
        self.metrics = metrics
    def fit(self, X, y, max_iter, learning_rate):
        for i in range(max_iter):
            activation = X
            for layer in self.layers:
                activation = layer.forward(activation)

            # for output layer
            if self.loss == mse or self.loss == mae:
                delta = self.loss_d(y, self.layers[-1].A) * self.layers[-1].activation_d(self.layers[-1].Z)
            else:
                delta = self.loss_d(y, self.layers[-1].Z)

            output_layer = self.layers[-1]
            output_layer.dLdW = output_layer.input.T @ delta
            output_layer.dLdb = np.sum(delta, axis=0, keepdims=True)

            output_layer.weights -= learning_rate * output_layer.dLdW
            output_layer.biases -= learning_rate * output_layer.dLdb

            # for hidden layers
            for layer in self.layers[::-1][1:]:
                delta = layer.backward(delta)
                layer.weights -= learning_rate * layer.dLdW
                layer.biases -= learning_rate * layer.dLdb

model = NN((2, ))
model.add(Layer(2, 3, relu, relu_d))
model.add(Layer(3, 2, leaky_relu, leaky_relu_d))
model.add(Layer(2, 1, sigmoid, sigmoid_d))

model.compile(binary_cross_entropy, sigmoid_BCE_d, accuracy)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

model.fit(X, y, max_iter=2, learning_rate=0.001)