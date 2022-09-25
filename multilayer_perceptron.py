from __future__ import nested_scopes
from ast import main
from os import W_OK
import numpy as np


class MultilayerPerceptronClassifier():
    def __init__(self, X, y, net_shape=(2, 2, 1)):
        self.X = X
        self.y = y
        self.N = len(X)
        self.L = len(net_shape)
        self.n_classes = np.unique(self.y[0])
        self.n_feats = self.X.shape[0]
        self.net_shape = net_shape
        self.weights = self.init_weights()
        self.net_outf = self.init_input_vectors()
        self.net_outv = self.init_neuron_outputs()
        self.deltas = self.init_deltas()

    def train(self, lr, epochs=10):
        for e in range(epochs):
            self.forward_pass()
            self.backpropagation()
            self.update_weights()
    
    def predict(self, X):
        preds = []
        for i in range(self.N):
            for r in range(1, self.L):
                v = np.append(np.zeros(self.net_shape[r]), 1) # Create vector to hold neuron outputs + 1 for extended
                for k in range(self.net_shape[r]):
                    w = self.get_weight(r, k)
                    neuron_out = w[:-1].T.dot(self.net_outf[r-1][i]) + w[-1]
                    v[k] = neuron_out
                v_f = list(map(self.activation, v[:-1]))
            preds.append(v_f)
        return preds

    def forward_pass(self):
        """ Hidden layer neurons creates mostly the same result? """
        for i in range(self.N):
            for r in range(1, self.L):
                v = np.append(np.zeros(self.net_shape[r]), 1) # Create vector to hold neuron outputs + 1 for extended
                for k in range(self.net_shape[r]):
                    w = self.get_weight(r, k)
                    neuron_out = w[:-1].T.dot(self.net_outf[r-1][i]) + w[-1]
                    v[k] = neuron_out
                v_f = list(map(self.activation, v[:-1]))
        
                self.net_outv[r][i] = v[:-1]  # Store the acutal output.
                self.net_outf[r][i] = v_f  # Store ativation output of neurons, they are the next layer inputs.

    def backpropagation(self):
        for i in range(self.N):
            for r in range(self.L-1, -1, -1):
                for k in range(self.net_shape[r]):
                    if r == self.L - 1: # Layer L
                        #print(self.net_outf)
                        delta = self.net_outf[r][i][k] * (1 - self.net_outf[r][i][k])
                        delta = (self.net_outf[r][i][k] - self.y[i]) * delta   # y[i] will not work for multiple output neurons
                        self.deltas[r][i][k] = delta
                    else:
                        delta = self.net_outf[r][i][k] * (1 - self.net_outf[r][i][k])
                        summa = 0
                        for prev_k in range(self.net_shape[r+1]):
                            w_prev_layer = self.get_weight(r+1, prev_k, k)
                            delta_prev = self.deltas[r+1][i][prev_k]
                            summa += w_prev_layer * delta_prev
                        self.deltas[r][i][k] = delta*summa


    def update_weights(self, lr=0.02):
        for r in range(1, self.L):
            for j in range(self.net_shape[r]-1):
                dr = np.tile(self.deltas[r][:, j], len(self.net_outf[r-1]))
                df = np.c_[self.net_outf[r-1], np.ones(len(self.net_outf[r-1]))]
                summation = np.sum((self.net_outf[r-1], np.c_[self.deltas[r][:, j], np.ones(len(self.deltas[r][:, j]))]))
                self.weights[r][j] -= lr * summation
                
    
    def activation(self, x, type="sigmoid"):
        if type == "sigmoid":
            res = 1/(1+np.exp(-x))
        else:
            quit("Unknown activation function")

        return res

    def get_weight(self, layer, neuron, component=None):
        w = self.weights[layer-1][neuron]
        if component is not None:
            w = w[component]
        return w

    def init_weights(self):
        # Weights stored as:
        # W[LAYER][NEURON][WEIGHT i]
        w = [[] * 1 for r in self.net_shape[1:]]
        for r in range(len(self.net_shape[1:])):    # For each layer in the net
            for k in range(self.net_shape[1:][r]):      # For each neuron in the current layer
                w[r].append(np.append(np.random.uniform(0, 1, 2), 1))
        return w
        """ for i in w:
            print(i) """

    def extend_y(self):
        y_ext = np.zeros((self.N, self.N))
        for i in range(self.N):
            y_ext[i, i] = self.y[i][0]

        return y_ext


    def init_input_vectors(self):
        # [r][i][k]
        
        # Dont think this works for more layers
        # For 2 2 1 this creates 2 3 3 
        out = [np.ones((self.N, self.net_shape[r])) for r in range(len(self.net_shape))]
        out[0] = self.X
        #print(out[2][0])
        #quit()
        return out
    
    def init_neuron_outputs(self):
        return self.init_input_vectors()

    def init_deltas(self):
        # [r][i][k]
        # List of numpy arrays to hold deltas. Reverse it because it is in the wrong order for some reason.
        return [np.ones((self.N, r)) for r in self.net_shape]

def get_data(N):
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    m1_1 = np.array([0, 0])
    m1_2 = np.array([1, 1])
    m2_1 = np.array([0, 1])
    m2_2 = np.array([1, 0])

    x1_1 = np.random.multivariate_normal(m1_1, cov, N)
    x1_2 = np.random.multivariate_normal(m1_2, cov, N)
    x2_1 = np.random.multivariate_normal(m2_1, cov, N)
    x2_2 = np.random.multivariate_normal(m2_2, cov, N)

    x1_1 = (x1_1 - np.min(x1_1)) / (np.max(x1_1)- np.min(x1_1))
    x1_2 = (x1_1 - np.min(x1_1)) / (np.max(x1_1)- np.min(x1_1))
    x2_1 = (x1_1 - np.min(x1_1)) / (np.max(x1_1)- np.min(x1_1))
    x2_2 = (x1_1 - np.min(x1_1)) / (np.max(x1_1)- np.min(x1_1))

    y1 = np.zeros(len(x1_1) + len(x1_2))
    y2 = np.ones(len(x2_1) + len(x2_2))

    X = np.concatenate((x1_1, x1_2, x2_1, x2_2))
    y = np.concatenate((y1, y2))


    return X, y


if __name__ == "__main__":
    X, y = get_data(100)
    #X = np.c_[X, np.ones(len(X))]
    #y = np.c_[y, np.ones(len(y))]
    mlp = MultilayerPerceptronClassifier(X, y)
    mlp.train(lr=0.02, epochs=100)
    w = mlp.weights
    print(w)

    preds = np.array(mlp.predict(X))
    # print(preds)
    

