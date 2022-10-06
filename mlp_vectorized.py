import numpy as np
import matplotlib.pyplot as plt


class Layer():
    def __init__(self, weights, n_neurons:int, input=None, first_layer=False, last_layer=False):
        self.n_neurons = n_neurons
        self.weights = weights
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.input = input
        self.output = input if first_layer else []
        self.deltas = []
        self.deltas_next = []

    def predict(self, X):
        return np.round([self.activation(self.weights[i].dot(X.T)) for i in range(self.n_neurons)])

    def forward_pass(self, input):
        self.output.append(
            np.append(
                [self.activation(self.weights[i].T.dot(input)) for i in range(self.n_neurons)], 1)
            )
      
    def backward_propogate(self, input):
        self.output = np.array(self.output)
        
        if self.last_layer:
            for _ in range(self.n_neurons):
                self.deltas.append(
                    list((self.output * (1 - self.output[:, :-1]) * (self.output[:, :-1] - input))[:-1]))
        else:
            self.deltas.append(list((input * self.output[:, :-1] * (1 - self.output[:, :-1]))[:-1]))
        # Set the sum delta weight neede in the back propagation of the next layer
      
        for j in range(self.n_neurons):
            #print(np.sum(self.deltas * self.weights[:, j]))
            self.deltas_next.append(np.sum(self.deltas * self.weights[:, j]))

    def update_weights(self, lr, input):
        """ Gradient decent with momentum """
        #print(len(self.deltas), len(input))
        #print(np.sum(np.c_[self.deltas, np.ones(len(self.deltas))].T @ input))
        #w_change = -lr * np.sum(np.c_[self.deltas, np.ones(len(self.deltas))].T @ input)
        input = np.array(input)
        w_change = -lr * np.sum(self.deltas * input[:, :-1])
        self.weights += w_change
        self.deltas = []
        self.deltas_next = []


    def activation(self, x, type="sigmoid"):
        if type == "sigmoid":
            res = 1 / (1 + np.exp(-x))
        else:
            quit("Unknown activation function")

        return res

    def init_weights(self):
        return np.array([np.random.normal(-.5, .5, 3), 
                         np.random.normal(-.5, .5, 3)])
    
    def init_deltas(self):
        return np.ones((self.N, self.n_neurons))


class MultilayerPerceptronClassifier():
    def __init__(self, X, y, net_shape=(2, 2, 1)):
        self.X = X
        self.y = y
        self.N = len(X)
        self.L = len(net_shape)
        self.net_shape = net_shape
       
        self.input_layer = Layer(
            weights=None,
            n_neurons=2, input=self.X, first_layer=True)
        self.hidden_layer = Layer(
            weights=np.array([np.random.normal(-0.5, 0.5, 3), np.random.normal(-0.5, 0.5, 3)]),
            n_neurons=2)
        self.output_layer = Layer(
            weights=np.array([np.random.normal(-0.5, 0.5, 3)]),
            n_neurons=1, last_layer=True)

        self.layers = [self.input_layer, self.hidden_layer, self.output_layer]
    
    def predict(self, X):
        return self.output_layer.predict(X)

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
        for i in range(self.N):
            for r, layer in enumerate(self.layers[1:]):
                # Start at r + 1 which is initialy the first hidden layer.
                self.layers[r+1].forward_pass(self.layers[r].output[i])
              
    def back_propagation(self):
        for r in range(self.L-1, 0, -1):
            layer = self.layers[r]
            if layer.last_layer:
                layer.backward_propogate(self.y)
            else:
                layer.backward_propogate(self.layers[r+1].deltas_next[-1])

    def update_weights(self, a, lr):
        for r, layer in enumerate(self.layers[1:]):
            # Start at r + 1 which is initialy the first hidden layer.
            self.layers[r+1].update_weights(lr, self.layers[r].output)
            # Remove all values produced by the hidden and output layers
            if not self.layers[r].first_layer:
                self.layers[r].output = []
        self.layers[-1].output = []

    def get_cost(self, output):
        cost = 0
        for i in range(self.N):
            cost += (output[i][0] - self.y[i][0]) ** 2
        cost *= 0.5

        return cost

def get_data(N):
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    m1_1 = np.array([0, 0])
    m1_2 = np.array([0, 0])
    m2_1 = np.array([0, 1])
    m2_2 = np.array([0, 1])

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
    grid = np.c_[grid, np.ones(len(grid))]
    yhat = np.array(model.predict(grid))

    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap="Paired")
    
    plt.scatter(X[:len(X)//2, 0], X[:len(X)//2, 1], c='b')
    plt.scatter(X[len(X)//2:, 0], X[len(X)//2:, 1], c='r')
    plt.show()
    plt.savefig("Contour3.png")

if __name__ == "__main__":
    X, y = get_data(100)
    mlp = MultilayerPerceptronClassifier(X,y)
    for i in mlp.layers[1:]:
        print(i.weights)
    mlp.train(a=0.5, lr=0.001, epochs=100)

    predictions = mlp.predict(X)
    """ print(predictions)
    print(y[:, 0]) """
    print("Accuracy:", np.sum(y[:,0]==predictions)/len(y))
    #print(predictions)
    plot_boundaries(mlp, X)