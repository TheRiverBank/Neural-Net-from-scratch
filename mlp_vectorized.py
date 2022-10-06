import numpy as np
import matplotlib.pyplot as plt


class Layer():
    def __init__(self, n_neurons:int, input=None, first_layer=False, last_layer=False):
        self.n_neurons = n_neurons
        self.weights = self.init_weights()
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.input = input
        self.output = input if first_layer else []
        self.deltas = []
        self.deltas_next = []

    def forward_pass(self, input):
        self.output.append(
            np.append(
                [self.activation(self.weights[i].T.dot(input)) for i in range(self.n_neurons)],
                1
                )
            )
      
    def backward_propogate(self, input):
        delts = []
        if self.last_layer:
            for _ in range(self.n_neurons):
                self.deltas.append(list((self.output[-1] * (1 - self.output[-1]) * (self.output[-1] - input))[:-1]))
        else:
            self.deltas.append(list((input * self.output[-1] * (1 - self.output[-1]))[:-1]))
        # Set the sum delta weight neede in the back propagation of the next layer
      
        for j in range(self.n_neurons):
            #print(np.sum(self.deltas * self.weights[:, j]))
            
            self.deltas_next.append(np.sum(self.deltas * self.weights[:, j]))

    def update_weights(self, lr, input):
        """ Gradient decent with momentum """
        print(len(self.deltas), len(input))
        #print(np.sum(np.c_[self.deltas, np.ones(len(self.deltas))].T @ input))
        w_change = -lr * np.sum(np.c_[self.deltas, np.ones(len(self.deltas))].T @ input)
        #print(w_change)
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
       
        self.input_layer = Layer(n_neurons=2, input=self.X, first_layer=True)
        self.hidden_layer = Layer(n_neurons=2)
        self.output_layer = Layer(n_neurons=1, last_layer=True)

        self.layers = [self.input_layer, self.hidden_layer, self.output_layer]
     
    def train(self, a, lr, epochs=100):
        for e in range(epochs):
            print(e)
            for i in range(self.N):
                for r, layer in enumerate(self.layers[1:]):
                    # Start at r + 1 which is initialy the first hidden layer.
                    self.layers[r+1].forward_pass(self.layers[r].output[i])

            for i in range(self.N):
                for r in range(self.L-1, 0, -1):
                    layer = self.layers[r]
                    if layer.last_layer:
                        layer.backward_propogate(self.y[i])
                    else:
                        layer.backward_propogate(self.layers[r+1].deltas_next[-1])

            
            for r, layer in enumerate(self.layers[1:]):
                # Start at r + 1 which is initialy the first hidden layer.
                self.layers[r+1].update_weights(lr, self.layers[r].output[-self.N:])
        
                
        """  
        for i in self.layers[1].deltas:
            print(i)
        """
        for layer in self.layers:
            print(layer.weights) 
        


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
    X, y = get_data(10)

    mlp = MultilayerPerceptronClassifier(X,y)
    mlp.train(a=0.5, lr=0.01)
