from operator import xor
from types import new_class
from mlp_vectorized import MultilayerPerceptronClassifier
from data_util import *


def xor_test():
    N = 100
    X, y = get_XOR_data(N)
   
    class_labels = np.argmax(y[:], axis=1)
  
    mlp = MultilayerPerceptronClassifier(X, y, net_shape=(2, 2, 2))
    mlp.add_layer(2)
    mlp.add_layer(2)
    mlp.add_layer(2)

    mlp.train(a=0.1, lr=0.001, epochs=10000)

    predictions = mlp.predict(X,flat=True)

    print("Accuracy:", np.sum(class_labels==predictions)/len(y))
    plot_xor_boundaries(mlp, X)


def poly_test():
    N = 100
    X, y = get_poly_data(N)

    class_labels = np.argmax(y[:], axis=1)

    mlp = MultilayerPerceptronClassifier(X, y, net_shape=(2, 2, 2))
    mlp.add_layer(2)
    mlp.add_layer(2)
    mlp.add_layer(2)

    mlp.train(a=0.5, lr=0.01, epochs=10000)

    predictions = mlp.predict(X,flat=True)

    print("Accuracy:", np.sum(class_labels==predictions)/len(y))
    
    plot_poly_boundaries(mlp, X, N)



def multi_class_test():
    N = 100
    X, y = get_multi_class_data(9, N)

    class_labels = np.argmax(y[:], axis=1)

    mlp = MultilayerPerceptronClassifier(X, y, net_shape=(2, 20, 20, 9))
    mlp.add_layer(2)
    mlp.add_layer(20)
    mlp.add_layer(20)
    mlp.add_layer(9)

    mlp.train(a=0.3, lr=0.01, epochs=1000)

    predictions = mlp.predict(X,flat=True)

    print("Accuracy:", np.sum(class_labels==predictions)/len(y))
    
    plot_multi_class_boundaries(mlp, X, N, 9)

    
if __name__ == "__main__":
    #xor_test()
    #poly_test()
    multi_class_test()