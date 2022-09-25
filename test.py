from mlp import MLP
import numpy as np
import matplotlib.pyplot as plt


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

    yhat = np.array(model.predict(grid)[1])

    zz = yhat.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap="Paired")
    plt.scatter(X[:, 0], X[:, 1], c='b')


    plt.show()


if __name__ == '__main__':
    X, y = get_data(100)
    mlp = MLP(0.01, X, y)

    mlp.train(10)
    #plot_boundaries(mlp, X)
    w = mlp.weights
    print(w)
    #preds = mlp.predict(X)
    #print(np.array(preds))





