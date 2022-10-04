import numpy as np
import matplotlib.pyplot as plt


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
        self.prev_weights = self.init_prev_weights()
        self.net_outf = self.init_input_vectors()
        self.net_outv = self.init_neuron_outputs()
        self.deltas = self.init_deltas()
        self.prev_cost = None

        for i in self.net_outf:
            print(i)

    def init_weights(self):
        # W[LAYER][NEURON][WEIGHT i]
        w = [[] * 1 for r in self.net_shape[1:]]
        for r in range(len(self.net_shape[1:])):  # For each layer in the net
            for k in range(self.net_shape[1:][r]):  # For each neuron in the current layer
                w[r].append(np.random.normal(-.5, 0.5, 3))

        return w

    def init_prev_weights(self):
        w = [[] * 1 for r in self.net_shape[1:]]
        for r in range(len(self.net_shape[1:])):  # For each layer in the net
            for k in range(self.net_shape[1:][r]):  # For each neuron in the current layer
                w[r].append(np.zeros(3))
        return w

    def init_input_vectors(self):
        # [r][i][k]
        # Add one to the input vector to di the vectorized approach
        out = [np.ones((self.N, self.net_shape[r] + 1)) for r in range(len(self.net_shape))]
        out[0] = self.X

        return out

    def init_neuron_outputs(self):
        return self.init_input_vectors()

    def init_deltas(self):
        # [r][i][k]
        # Add one extra dimention for the vectorized approach
        return [np.ones((self.N, r + 1)) for r in self.net_shape[1:]]

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

    y1 = np.zeros(len(x1_1) + len(x1_2))
    y2 = np.ones(len(x2_1) + len(x2_2))

    X = np.concatenate((x1_1, x1_2, x2_1, x2_2))
    X = np.c_[X, np.ones(len(X))]
    
    y = np.concatenate((y1, y2))
    y = np.c_[y, np.ones(len(y))]

    return X, y


if __name__ == "__main__":
    X, y = get_data(100)

    mlp = MultilayerPerceptronClassifier(X,y)
