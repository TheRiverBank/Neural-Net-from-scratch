import numpy as np
import matplotlib.pyplot as plt


class MLP():
    def __init__(self, lr, X, y, layers=(2, 1)):
        self.lr = lr
        self.n_feats = X.shape[0]
        self.n_classes = len(np.unique(y))
        self.weights = None
        self.layers = layers
        self.X = np.c_[X, np.ones(len(X))]
        self.y = np.c_[y, np.ones(len(y))]

    def train(self, epochs=10):
        self.init_weights()
        print("INIT WEIGHTS: ", self.weights)
        cur_inp = self.X
        for _ in range(epochs):
            v, y_hat = self.forward_pass(cur_inp)
            deltas = self.back_propagate(y_hat)
            self.update_weights(deltas)

    def forward_pass(self, X):
        """
        Forward pass through the network.
        Returns the output prediction(s) of the network
        """
        v_all = [[]*1 for _ in range(len(self.layers))]
        y_hat_all = [[]*1 for _ in range(len(self.layers))]

        for x in X:
            layer_input = x
            for idx, r in enumerate(self.layers):
                cur_res = []
                layer_v = []
                for k in range(r):
                    w = self.weights[k + idx * len(self.layers)]

                    v = np.sum(w.T.dot(layer_input))
                    node_y_hat = self.activation(v, type="sigmoid")
                    cur_res.append(node_y_hat)
                    layer_v.append(v)

                if idx != len(self.layers) - 1:
                    cur_res.append(1)

                layer_input = np.array(cur_res)
                v_all[idx].append(layer_v)
                y_hat_all[idx].append(cur_res)

        return v_all, y_hat_all

    def back_propagate(self, y_hat):
        deltas = np.array([])
        for r in range(len(self.layers)-1, -1, -1):
            layer_deltas = []
            for i in range(len(self.X)):
                cur_y = y_hat[r][i][0]
                if r == len(self.layers)-1:
                    error = np.sum(cur_y - self.y[i])
                    # Error times sigmoid derivative
                    delta = error * (cur_y * (1 - cur_y))
                    layer_deltas.append(delta)
                else:
                    for k in range(self.layers[r]):
                        delta = (np.sum(self.weights[r, k] * deltas[r-1]) *
                                (y_hat[r-1][i][0] * (1 - y_hat[r-1][i][0])))
                        layer_deltas.append(delta)
            deltas = np.append(deltas, layer_deltas)

        return deltas[-len(self.X):]

    def update_weights(self, deltas):
        deltas = np.c_[deltas, np.ones(len(deltas))]
        self.weights += (-self.lr * (np.sum(deltas*self.y)))

    def init_weights(self):
        w_lst = []
        for i in range(self.n_feats):
            w_lst.append(np.random.uniform(0, 1, 3))
        self.weights = np.array(w_lst)

    def activation(self, x, type="sigmoid"):
        if type == "sigmoid":
            res = 1/(1+np.exp(-x))
        else:
            quit("Unknown activation function")

        return res

    def compute_error(self, y_hat, type="mss"):
        if type == "mss":
            pass
        else:
            quit("Unknown error function")

    def predict(self, X):
        _, y_hat_all = self.forward_pass(np.c_[X, np.ones(len(X))])
        return y_hat_all

