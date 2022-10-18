
import numpy as np
import matplotlib.pyplot as plt


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


def get_multi_class_data(n_classes, N):
    X = np.zeros((N*n_classes, 2))
    Y = np.zeros((N*n_classes))
    
    cov = np.diag(np.full(2, 0.001))

    for i in range(n_classes):
        m = np.array([i, i%2])
        x = np.random.multivariate_normal(m, cov, N)
        y = np.ones(N) * i
       
        X[N*i:N*i+N] = x
        Y[N*i:N*i+N] = y

    X = np.c_[X, np.ones(len(X))]
    Y = np.c_[Y, np.ones(len(Y))]

    return X, Y



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


from matplotlib.pyplot import cm
def plot_poly_boundaries(model, X, N):
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


    color = cm.rainbow(np.linspace(0, 1, len(X)//N))

    for i in range(len(X)//N):
        plt.scatter(X[N*i:N*i+N:, 0], X[N*i:N*i+N:, 1], c=color[i])
    plt.show()
    plt.savefig("Contour3.png")