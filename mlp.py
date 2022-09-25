import numpy as np
import matplotlib.pyplot as plt


class MLP():
    def __init__(self, lr, X, y, layers=(2, 1)):
        self.lr = lr
        self.n_feats = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.weights = None
        self.layers = layers
        self.X = np.c_[X, np.ones(len(X))]
        self.y = np.c_[y, np.ones(len(y))]

    def train(self, epochs=1000):
        self.init_weights()
        #print("INIT WEIGHTS: ", self.weights)
        cur_inp = self.X
        for _ in range(epochs):
            v, y_hat = self.forward_pass(cur_inp)
            deltas = self.back_propagate(y_hat)
            self.update_weights(deltas, y_hat)

 


    def forward_pass(self, X):
        """
        Forward pass through the network.
        Returns the output prediction(s) of the network
        """
        v_all = []
        y_hat_all = []

        for x in X:
            layer_input = x
            # v_x = [[]*1 for _ in range(len(self.layers))]
            y_hat_x = [[]*1 for _ in range(len(self.layers))]
            for idx, r in enumerate(self.layers):
                cur_res = []
                layer_v = []
                for k in range(r):
                    w = self.get_weight(idx, k)
                    if w.shape == (1, 3):
                        w = w[0]
                 
                    v = w.T.dot(layer_input)
                    node_y_hat = self.activation(v, type="sigmoid")
                    #print(node_y_hat, v, layer_input)
                   
                    cur_res.append(int(node_y_hat))
                    layer_v.append(v)

                cur_res.append(1)

                layer_input = np.array(cur_res)
                # v_all[idx].append(layer_v)
                y_hat_x[idx].append(cur_res)
            y_hat_all.append(y_hat_x)
       
        return v_all, y_hat_all

    def back_propagate(self, y_hat):
        deltas = []
        for x in range(len(self.X)-1, -1, -1):
            delta_x = []
            for r in range(len(self.layers)-1, -1, -1):
                layer_deltas = []
                #print(y_hat[x])
                if r == len(self.layers)-1:
                    for k in range(self.layers[r]):
                        cur_y_hat = y_hat[x][r][k][0]
                        error = (cur_y_hat - self.y[x][0])  # y[x] will not work for multiple neurons in last layer
                        d = error * (cur_y_hat * (1 - cur_y_hat))
                        #print(d)
                        layer_deltas.append(d)
                else:
                    for j in range(self.layers[r]):
                        sum_prev = 0
                        for k in range(self.layers[r+1]):
                            w = self.get_weight(r + 1, k)
                            if w.shape == (1, 3):
                                w = w[0]
                            w_prev = w
                            d_prev = delta_x[r-1][k]
                            sum_prev += w_prev * d_prev
                        #print(y_hat[x][r][0][j], x, r, j)
                        cur_y_hat = y_hat[x][r][0][j]
                        d = sum_prev * (cur_y_hat * (1 - cur_y_hat))
                        #print(d, cur_y_hat)
                        layer_deltas.append(d)
                layer_deltas.append(1)
                delta_x.append(layer_deltas)
            deltas.append(delta_x)
        return deltas

    def get_delta(self, deltas, layer, neuron):
        return deltas[layer][neuron]

    def update_weights(self, deltas, y_hats):
        #deltas = np.c_[deltas, np.ones(len(deltas))]
        for r in range(len(self.layers)):
            for j in range(self.layers[r]):
                w_old = self.get_weight(r, j)
                w_change = 0
                for i in range(len(self.X)):
                    y_hat = self.X[i] if r == 0 else y_hats[i][r-1]
                    print(y_hat)
                    w_change += deltas[i][r][j] * np.array(y_hat)
                w_new = w_old - (self.lr * w_change)
                self.set_weight(r, j, w_new)

    def init_weights(self):
        w_lst = []
        for idx_r, r in enumerate(self.layers):
            layer_weights = []
            for k in range(r):
                n_inputs = self.n_feats if (idx_r == 0) else self.layers[idx_r-1]
                layer_weights.append(np.array([*np.random.uniform(0, 1, n_inputs), 1]))
            w_lst.append(layer_weights)

        self.weights = w_lst

    def get_weight(self, layer, neuron):
        return self.weights[layer][neuron]

    def set_weight(self, layer, neuron, w):
        self.weights[layer][neuron] = w

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
        X = np.c_[X, np.ones(len(X))]
        preds = []
        for x in X:
            layer_input = x
            output_neuron_vals = []
            for idx, r in enumerate(self.layers):
                for k in range(r):
                    w = self.get_weight(idx, k)
                    v = np.sum(w.dot(layer_input))
                    node_y_hat = self.activation(v, type="sigmoid")
                    if idx == len(self.layers)-1:
                        output_neuron_vals.append(node_y_hat)
            preds.append(output_neuron_vals)

        return preds

