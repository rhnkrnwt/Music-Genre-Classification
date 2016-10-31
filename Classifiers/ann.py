import numpy as np
from scipy.special import expit
from sklearn import preprocessing
from functools import reduce
from total_arrange import get_data

if __name__ == '__main__':
    # get training and test data
    A1, Y, Ate, Yte = get_data('dataset3.csv', 5, 'hot')
    A1 = preprocessing.scale(A1)
    Ate = preprocessing.scale(Ate)


    for k in range(9, 10):
        np.random.seed(7)
        eta = 0.01
        nh = [20] * 11
        W = [0] * 11
        G = [0] * 11
        Ao = [0] * 11
        d = [0] * 11

        # h1(hidden) x m(input)
        W[0] = 2 * np.random.random((nh[0], 136)) - 1
        G[0] = np.zeros((nh[0], 136))

        for i in range(1, 10):
            W[i] = 2 * np.random.random((nh[i], nh[i - 1])) - 1
            G[i] = np.zeros((nh[i], nh[i - 1]))


        W[10] = 2 * np.random.random((4, nh[9])) - 1
        G[10] = np.zeros((4, nh[9]))

        for i in range(1, 10000):
            # expit(x) = 1 / (1 + exp(-x))
            eta = 1 / i
            # A2
            Ao[0] = expit(np.dot(W[0], A1.T))

            for j in range(1, 11):
                Ao[j] = expit(np.dot(W[j], Ao[j - 1]))


            d[10] = Ao[10] - Y.T

            for j in range(9, -1, -1):
                d[j] = np.dot(W[j + 1].T, d[j + 1]) * (Ao[j] * (1 - Ao[j]))

            for j in range(10, 0, -1):
                # print(G[j].shape)
                # print(d[j].shape)
                # print(Ao[j-1].shape)
                G[j] = G[j] + np.dot(d[j], Ao[j - 1].T)
            G[0] = G[0] + np.dot(d[0], A1)

            for j in range(1, 11):
                W[j] = W[j] - (eta * G[j])

        """
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
        """
        diff = np.rint(Ao[10].T) - Y
        count_correct = 0
        print(diff)
        for i in diff:
            if np.array_equal(i, [0, 0, 0, 0]):
                count_correct += 1
        print(count_correct)
        """
        print('[')
        for i in range(W2.shape[0]):
            print('[',end=' ')
            for j in range(W2.shape[1]):
                print(W2[i, j], end=' ')
            print('],', end=' ')
        print(']')
        """
