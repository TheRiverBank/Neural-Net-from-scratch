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
         
    def predict(self, X):
        out = [np.ones((len(X), self.net_shape[r])) for r in range(len(self.net_shape))]
        out[0] = X
        preds = []
        for i in range(len(X)):
            for r in range(1, self.L):
                v = np.append(np.zeros(self.net_shape[r]), 1)  # Create vector to hold neuron outputs + 1 for extended
                for k in range(self.net_shape[r]):
                    w = self.get_weight(r, k)
                    neuron_out = w[:-1].T.dot(out[r - 1][i]) + w[-1]
                    v[k] = neuron_out
                v_f = list(map(self.activation, v[:-1]))
                out[r][i] = v_f
            preds.append(v_f)  # Last v_f is the output of the last neurons
        return preds

    def train(self, lr, a, epochs=10):
        for e in range(epochs):

            self.forward_pass()
            self.backpropagation()
            self.update_weights(lr, a)
            cost = self.print_cost()
            print(cost, e)


    def forward_pass(self):
        for i in range(self.N):
            for r in range(1, self.L):
                v = np.zeros(self.net_shape[r])  # Create vector to hold neuron outputs + 1 for extended
                for k in range(self.net_shape[r]):
                    w = self.get_weight(r, k)
                    neuron_out = w[:-1].T.dot(self.net_outf[r - 1][i]) + w[-1]
                    v[k] = neuron_out
                v_f = list(map(self.activation, v))

                self.net_outv[r][i] = v  # Store the acutal output.
                self.net_outf[r][i] = v_f  # Store ativation output of neurons, they are the next layer inputs.

    def backpropagation(self):
        for i in range(self.N):
            for r in range(self.L - 1, 0, -1):
                for j in range(self.net_shape[r]):
                    if r == self.L - 1:  # Layer L
                        delta = self.net_outf[r][i][j] * (1 - self.net_outf[r][i][j])
                        delta = (self.net_outf[r][i][j] - self.y[
                            i]) * delta  # y[i] will not work for multiple output neurons
                        self.deltas[r - 1][i][j] = delta
                    else:
                        delta = self.net_outf[r][i][j] * (1 - self.net_outf[r][i][j])
                        prev_sum = 0
                        for prev_k in range(self.net_shape[r + 1]):
                            w_prev_layer = self.get_weight(r + 1, prev_k, j)
                            delta_prev = self.deltas[r][i][prev_k]
                            prev_sum += w_prev_layer * delta_prev
                        self.deltas[r - 1][i][j] = delta * prev_sum

    def update_weights(self, lr, a):
        """ Gradient decent with momentum """
        for r in range(0, self.L - 1):
            for j in range(0, self.net_shape[r + 1]):
                delta = self.deltas[r][:, j]
                outputs = self.net_outf[r - 1]
                w_change = (a*self.prev_weights[r][j]) - lr * np.sum(delta @ outputs)
                self.prev_weights[r][j] = w_change # Store current change and use it as momentum next iteration.
                self.weights[r][j] += w_change

    def print_cost(self):
        cost = 0
        for i in range(self.N):
            cost += (self.net_outf[2][i] - self.y[i]) ** 2
        cost * 0.5

        return cost[0]

    def plot_updates(self, xx, yy, grid, epoch, cost):
        yhat = np.array(self.predict(grid))

        # print(yhat)
        yhat = np.array([round(x[0]) for x in yhat])

        zz = yhat.reshape(xx.shape)

        plt.contourf(xx, yy, zz, cmap="Paired")
        plt.scatter(X[:, 0], X[:, 1], c='b')
        plt.pause(0.0001)
        plt.clf()
        plt.plot()


    def activation(self, x, type="sigmoid"):
        if type == "sigmoid":
            res = 1 / (1 + np.exp(-x))
        else:
            quit("Unknown activation function")

        return res

    def get_weight(self, layer, neuron, component=None):
        w = self.weights[layer - 1][neuron]
        if component is not None:
            w = w[component]

        return w

    def init_weights(self):
        # Weights stored as:
        # W[LAYER][NEURON][WEIGHT i]
        w = [[] * 1 for r in self.net_shape[1:]]
        for r in range(len(self.net_shape[1:])):  # For each layer in the net
            for k in range(self.net_shape[1:][r]):  # For each neuron in the current layer
                w[r].append(np.random.normal(0, 0.5, 3))

        return w

    def init_prev_weights(self):
        w = [[] * 1 for r in self.net_shape[1:]]
        for r in range(len(self.net_shape[1:])):  # For each layer in the net
            for k in range(self.net_shape[1:][r]):  # For each neuron in the current layer
                w[r].append(np.append(np.zeros(2), 1))
        return w

    def extend_y(self):
        y_ext = np.zeros((self.N, self.N))
        for i in range(self.N):
            y_ext[i, i] = self.y[i][0]

        return y_ext

    def init_input_vectors(self):
        # [r][i][k]
        out = [np.ones((self.N, self.net_shape[r])) for r in range(len(self.net_shape))]
        out[0] = self.X

        return out

    def init_neuron_outputs(self):
        return self.init_input_vectors()

    def init_deltas(self):
        # [r][i][k]
        return [np.ones((self.N, r)) for r in self.net_shape[1:]]


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
    y = np.concatenate((y1, y2))

    return X, y


def plot_boundaries(model, X):
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))

    yhat = np.array(model.predict(grid))

    yhat = np.array([round(x[0]) for x in yhat])

    zz = yhat.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap="Paired")

    
    plt.scatter(X[:len(X)//2, 0], X[:len(X)//2, 1], c='b')
    plt.scatter(X[len(X)//2:, 0], X[len(X)//2:, 1], c='r')
    plt.show()
    plt.savefig("Contour3.png")


if __name__ == "__main__":
    X, y = get_data(1000)
    # X = np.c_[X, np.ones(len(X))]
    # y = np.c_[y, np.ones(len(y))]
    mlp = MultilayerPerceptronClassifier(X, y)
    mlp.train(lr=0.01, a=0.5, epochs=2000)
    w = mlp.weights
    ([print(x) for x in sum(w, [])])

    preds = np.array(mlp.predict(X))
    y_hat = [round(x[0]) for x in preds]
    
    print(np.sum(y==y_hat)/len(preds))

    plot_boundaries(mlp, X)



