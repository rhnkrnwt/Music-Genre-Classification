import numpy as np
from scipy.special import expit
from functools import reduce
from arrange import get_data


def tr(x, y):
    a = np.array([[0, 0, 0, 0]])
    if y.all() == a.all():
        return x + 1
    return x

if __name__ == '__main__':
    # get training and test data
    A1, Y, Ate, Yte = get_data()

    for j in range(5000, 5001):
        np.random.seed(1)
        eta = 0.1
        nh = j + 1
        # h(hidden) x m(input)
        W1 = 2 * np.random.random((nh, 11250)) - 1
        G1 = np.zeros((nh, 11250))

        # t(output) x h
        W2 = 2 * np.random.random((4, nh)) - 1
        G2 = np.zeros((4, nh))

        for i in range(1, 100):
            # expit(x) = 1 / (1 + exp(-x))
            eta = 1 / i
            A2 = expit(np.dot(W1, A1.T))
            A3 = expit(np.dot(W2, A2))

            d3 = A3 - Y.T
            d2 = np.dot(W2.T, d3) * (A2 * (1 - A2))

            G2 = G2 + np.dot(d3, A2.T)
            G1 = G1 + np.dot(d2, A1)

            W2 = W2 - (eta * G2)
            W1 = W1 - (eta * G1)

        corr = reduce(tr, np.rint(A3.T) - Y, 0)
        ratio = corr / Y.shape[0]

        print("Number of hidden units: {0}".format(nh))
        print("Training accuracy: {0}%".format( ratio * 100))

        # running the test data on the neural network
        A2te = expit(np.dot(W1, Ate.T))
        A3te = expit(np.dot(W2, A2te))
        corr = reduce(tr, np.rint(A3te.T) - Yte, 0)
        ratio = corr / Yte.shape[0]

        print("Test accuracy: {0}%\n".format( ratio * 100))

        print(np.rint(A3te.T))
        """
        print('[')
        for i in range(W2.shape[0]):
            print('[',end=' ')
            for j in range(W2.shape[1]):
                print(W2[i, j], end=' ')
            print('],', end=' ')
        print(']')
        """
