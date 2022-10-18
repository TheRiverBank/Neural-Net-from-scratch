import numpy as np
import matplotlib.pyplot as plt
from data_util import *


class Layer():
    def __init__(self, weights, n_neurons, net_input=None, first_layer=False):
        self.n_neurons = n_neurons
        self.weights = weights
        self.first_layer = first_layer
        self.net_input = net_input
        self.output = net_input if first_layer else []
        self.deltas = []
        self.deltas_next = []

    def predict(self, X):
        return np.round([self.activation(self.weights[i].T.dot(X.T)) for i in range(self.n_neurons)])

    def forward_pass(self, net_input):
        outf = np.ones((len(net_input), self.n_neurons + 1))
        for k in range(self.n_neurons):
            outf[:, k] = np.array([self.activation(self.weights[k].dot(np.array(net_input).T))])
        self.output = outf
       
    def backward_propogate(self, net_input, last_layer=False):
        if last_layer:
            for _ in range(self.n_neurons):
                self.deltas.append(
                    (self.output[:, :-1] * (1 - self.output[:, :-1]) * 
                    (self.output[:, :-1] - net_input[:, :-1])))
            self.deltas_next = np.column_stack(((np.array(self.deltas[0])*self.weights[0, 0]), (np.array(self.deltas[0])*self.weights[0, 1])))
        else:
            print(net_input.shape, self.output[:, :-1].shape)
            self.deltas = net_input @ (self.output[:, :-1] * (1 - self.output[:, :-1]))

    def update_weights(self, lr, net_input):
        """ Gradient decent with momentum """
        net_input = np.array(net_input)

        for k in range(self.n_neurons):
            if self.n_neurons == 1:
                w_change = -lr * self.deltas[k].T @ net_input
                self.weights[k] += w_change[0]
            else:
                w_change = -lr * self.deltas[:, k].T @ net_input
                self.weights[k, :] += w_change

        self.deltas = []
        self.deltas_next = []

    def activation(self, x, type="sigmoid"):
        if type == "sigmoid":
            res = 1 / (1 + np.exp(-x))
        else:
            quit("Unknown activation function")

        return res
    
    def init_deltas(self):
        return np.ones((self.N, self.n_neurons + 1))


class MultilayerPerceptronClassifier():
    def __init__(self, X, y, net_shape=(2,2,1)):
        self.X = X
        self.y = y
        self.N = len(X)
        self.L = len(net_shape)
        self.n_features = X.shape[1]
        self.net_shape = net_shape
       
        self.layers = []
    
    def predict(self, X):
        self.layers[0].output = X
        for r, layer in enumerate(self.layers[1:]):
            self.layers[r+1].forward_pass(self.layers[r].output)
        
        preds = np.round(self.layers[-1].output)

        return preds

    def train(self, a, lr, epochs=1000):
        for e in range(epochs):
            self.forward_pass()
            cost = self.get_cost(self.layers[-1].output)
            print(f"Epoch:{e}\tCost: {cost:.4f}")
            self.back_propagation()
            self.update_weights(a, lr)

        for layer in self.layers[1:]:
            print(layer.weights) 

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
            self.layers[r+1].update_weights(lr, self.layers[r].output)
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

if __name__ == "__main__":
    #X, y = get_test_data(100)
    X, y = get_poly_data(100)

    # net_input_layer = Layer(
    #         weights=None,
    #         n_neurons=2, net_input=X, first_layer=True)
    # hidden_layer = Layer(
    #     weights=np.array([[0.2, 0.5, 1.5], [0.1, 0.2, 0.3]]),
    #     n_neurons=2)
    # output_layer = Layer(
    #     weights=np.array([[0.6, -0.2, 0.4]]),
    #     n_neurons=1, last_layer=True)

    mlp = MultilayerPerceptronClassifier(X,y)
    mlp.add_layer(2)
    mlp.add_layer(3)
    mlp.add_layer(1)
    for i in mlp.layers[1:]:
        print(i.weights)
    mlp.train(a=0.5, lr=0.01, epochs=10000)

    predictions = mlp.predict(X)
    print("Accuracy:", np.sum(y[:,0]==predictions[:, 0])/len(y))
    print(predictions)
    #plot_boundaries(mlp, X)
    plot_poly_boundaries(mlp, X)