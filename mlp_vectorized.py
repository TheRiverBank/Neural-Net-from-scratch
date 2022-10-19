import numpy as np
import matplotlib.pyplot as plt
from data_util import *


class Layer():
    def __init__(self, weights, n_neurons, net_input=None, first_layer=False):
        self.n_neurons = n_neurons
        self.weights = weights
        self.previous_weights = np.zeros_like(weights)
        self.first_layer = first_layer
        self.net_input = net_input
        self.output = net_input if first_layer else []
        self.deltas = []
        self.deltas_next = []

    def forward_pass(self, net_input):
        outf = np.ones((len(net_input), self.n_neurons + 1))
        for k in range(self.n_neurons):
            outf[:, k] = np.array([self.activation(self.weights[k].dot(np.array(net_input).T))])
        self.output = outf
    
    def backward_propogate(self, net_input, last_layer=False):
        if last_layer:
            for _ in range(self.n_neurons):
                self.deltas = (
                    (self.output[:, :-1] * (1 - self.output[:, :-1]) * 
                    (self.output[:, :-1] - net_input[:, :-1])))
        else:
            self.deltas = net_input * (self.output[:, :-1] * (1 - self.output[:, :-1]))
        
        self.compute_next_delta(len(net_input))

    def update_weights(self, alpha, lr, net_input):
        """ Gradient decent with momentum """
        net_input = np.array(net_input)
        for k in range(self.n_neurons):
            w_change = -lr * self.deltas[:, k].T @ net_input
            w_old = self.previous_weights[k]
            self.weights[k, :] += w_change + w_old*alpha
            self.previous_weights[k] = w_change
        self.deltas = []
        self.deltas_next = []

    def compute_next_delta(self, n):
        d = np.zeros((n, len(self.weights[0])-1))
        
        for k in range(self.n_neurons):
            for j in range(len(self.weights[0])-1):
                d[:, j] += self.deltas[:, k] * self.weights[k, j]
        self.deltas_next = d

    def activation(self, x, type="sigmoid"):
        if type == "sigmoid":
            res = 1 / (1 + np.exp(-x))
        else:
            quit("Unknown activation function")

        return res
    
    def init_deltas(self):
        return np.ones((self.N, self.n_neurons + 1))


class MultilayerPerceptronClassifier():
    def __init__(self, X, y, net_shape):
        self.X = X
        self.y = y
        self.N = len(X)
        self.L = len(net_shape)
        self.n_features = X.shape[1]
        self.net_shape = net_shape
       
        self.layers = []
    
    def predict(self, X, flat=False):
        """
        :param X: numpy ndarray
        :param flat: bool, returns class labels if true or 0/1 matrix of dim num classes if false.
        """
        self.layers[0].output = X
        for r, layer in enumerate(self.layers[1:]):
            self.layers[r+1].forward_pass(self.layers[r].output)
        
        out = self.layers[-1].output[:, :-1]
        preds = np.zeros_like(out)
        preds[np.arange(len(out)), out.argmax(1)] = 1
       
        class_labels = np.argmax(self.layers[-1].output[:, :-1], axis=1)
        
        if flat:
            return class_labels
        return preds

    def train(self, a, lr, epochs=1000):
        for e in range(epochs):
            self.forward_pass()
            cost = self.get_cost(self.layers[-1].output)
            print(f"Epoch:{e}\tCost: {cost:.4f}")
            self.back_propagation()
            self.update_weights(a, lr)

    def forward_pass(self):
        for r, layer in enumerate(self.layers[1:]):
            self.layers[r+1].forward_pass(self.layers[r].output)
              
    def back_propagation(self):
        for r in range(self.L-1, 0, -1):
            layer = self.layers[r]
            if r == self.L - 1:
                layer.backward_propogate(self.y, last_layer=True)
            else:
                layer.backward_propogate(self.layers[r+1].deltas_next)

    def update_weights(self, a, lr):
        for r, layer in enumerate(self.layers[1:]):
            # Start at r + 1 which is initialy the first hidden layer.
            self.layers[r+1].update_weights(a, lr, self.layers[r].output)
            # Remove all values produced by the hidden and output layers
            if not self.layers[r].first_layer:
                self.layers[r].output = []

    def get_cost(self, output):
        cost = 0
        for i in range(self.N):
            cost += (output[i][0] - self.y[i][0]) ** 2
        cost /= self.N

        return cost

    def add_layer(self, n_neurons):
        if len(self.layers) == 0:  # Input layer
            l = Layer(weights=None, net_input=self.X, n_neurons=n_neurons, first_layer=True)
        else:
            # Give all neurons number of features weight components
            # The number of weights for this layer is the number of neurons in the previous layer. 
            # + 1 for each bias in each of the previous nodes
            w = np.random.normal(0, 0.5, self.layers[-1].n_neurons*n_neurons + n_neurons).reshape(n_neurons, self.layers[-1].n_neurons + 1)
            l = Layer(weights=w, n_neurons=n_neurons)
        self.layers.append(l)
        print(l.weights)
