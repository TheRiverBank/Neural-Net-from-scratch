import numpy as np
import matplotlib.pyplot as plt


class Layer():
    def __init__(self, weights, n_neurons, net_input=None, first_layer=False, last_layer=False):
        self.n_neurons = n_neurons
        self.weights = weights
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.net_input = net_input
        self.output = net_input if first_layer else []
        self.deltas = []
        self.deltas_next = []

    def predict(self, X):
        print(self.weights[0].T.dot(X[0].T), X[0])
        return np.round([self.activation(self.weights[i].T.dot(X.T)) for i in range(self.n_neurons)])

    def forward_pass(self, net_input):
        outf = np.ones((len(net_input), self.n_neurons + 1))
        for k in range(self.n_neurons):
            outf[:, k] = np.array([self.activation(self.weights[k].dot(np.array(net_input).T))])
        self.output = outf
       
    def backward_propogate(self, net_input):
        if self.last_layer:
            for _ in range(self.n_neurons):
                self.deltas.append(
                    (self.output[:, :-1] * (1 - self.output[:, :-1]) * 
                    (self.output[:, :-1] - net_input[:, :-1])))
            self.deltas_next = np.column_stack(((np.array(self.deltas[0])*self.weights[0, 0]), (np.array(self.deltas[0])*self.weights[0, 1])))
        else:
            self.deltas = net_input * (self.output[:, :-1] * (1 - self.output[:, :-1]))

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

    def init_weights(self):
        return np.array([-0.5, 0.5, 1.5], 
                        [0.2, 0.1, 1.5])
    
    def init_deltas(self):
        return np.ones((self.N, self.n_neurons + 1))


class MultilayerPerceptronClassifier():
    def __init__(self, X, y, layers, net_shape=(2,2,1)):
        self.X = X
        self.y = y
        self.N = len(X)
        self.L = len(net_shape)
        self.net_shape = net_shape
       
        self.layers = layers
    
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
            if layer.last_layer:
                layer.backward_propogate(self.y)
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


def get_XOR_data(N):
    cov = np.array([[0.001, 0.0], [0.0, 0.001]])
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


def get_poly_data(N):
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    m1_1 = np.array([0, 1])
    m1_2 = np.array([0.8, 1])
    m1_3 = np.array([1, 0.8])
    m1_4 = np.array([1, 0])
    m2 = np.array([0.6, 0.6])

    x1_1 = np.random.multivariate_normal(m1_1, cov, N)
    x1_2 = np.random.multivariate_normal(m1_2, cov, N)
    x1_3 = np.random.multivariate_normal(m1_3, cov, N)
    x1_4 = np.random.multivariate_normal(m1_4, cov, N)
    x2 = np.random.multivariate_normal(m2, cov, N)

    y1 = np.zeros(N*4)
    y2 = np.ones(N)

    X = np.concatenate((x1_1, x1_2, x1_3, x1_4, x2))
    X = np.c_[X, np.ones(len(X))]
    
    y = np.concatenate((y1, y2))
    y = np.c_[y, np.ones(len(y))]

    return X, y


def get_test_data(N):
    X = np.array([[0, 0, 1], [0, 0.5, 1], [1, 1, 1], [1, 1.5, 1]])
    y = np.array([[0, 1], [0, 1], [1, 1], [1, 1]])

    return X, y


def plot_xor_boundaries(model, X):
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
    # horizontal stack vectors to create x1,x2 net_input for the model
    grid = np.hstack((r1, r2))
    grid = np.c_[grid, np.ones(len(grid))]
    yhat = np.array(model.predict(grid))[:, 0]
    
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap="Paired")
    
    plt.scatter(X[:len(X)//2, 0], X[:len(X)//2, 1], c='b')
    plt.scatter(X[len(X)//2:, 0], X[len(X)//2:, 1], c='r')
    plt.show()
    plt.savefig("Contour3.png")


def plot_poly_boundaries(model, X):
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
    # horizontal stack vectors to create x1,x2 net_input for the model
    grid = np.hstack((r1, r2))
    grid = np.c_[grid, np.ones(len(grid))]
    yhat = np.array(model.predict(grid))[:, 0]

    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap="Paired")
    
    plt.scatter(X[:400, 0], X[:400, 1], c='b')
    plt.scatter(X[400:, 0], X[400:, 1], c='r')
    plt.show()
    plt.savefig("Contour3.png")


if __name__ == "__main__":
    #X, y = get_test_data(100)
    X, y = get_poly_data(100)
    net_input_layer = Layer(
            weights=None,
            n_neurons=2, net_input=X, first_layer=True)
    hidden_layer = Layer(
        weights=np.array([[0.2, 0.5, 1.5], [0.1, 0.2, 0.3]]),
        n_neurons=2)
    output_layer = Layer(
        weights=np.array([[0.6, -0.2, 0.4]]),
        n_neurons=1, last_layer=True)

    layers = [net_input_layer, hidden_layer, output_layer]

    mlp = MultilayerPerceptronClassifier(X,y,layers=layers)

    for i in mlp.layers[1:]:
        print(i.weights)
    mlp.train(a=0.5, lr=0.01, epochs=20000)

    predictions = mlp.predict(X)
    print("Accuracy:", np.sum(y[:,0]==predictions[:, 0])/len(y))
    print(predictions)
    #plot_boundaries(mlp, X)
    plot_poly_boundaries(mlp, X)