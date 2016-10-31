import numpy as np
from math import sqrt
from sklearn import mixture as gmm
from numpy import linalg as LA
def get_data():
    dat = np.loadtxt('../Data/dataset4.csv', delimiter = ',')
    classes = 4
    dat = np.array(dat[0:classes*100, :])
    labels = []
    for i in range(classes*100):
        v = int(i / 100)
        labels.append(v)
    labels = np.array(labels)
    p = np.random.permutation((len(dat)))

    return dat[p], labels[p]

def expected(X):
    n = X.shape[1]
    mu = X.mean(axis=1)
    X = X - mu[:,None]
    var = np.cov(X)
    return
    try:
        pdf = 1 / sqrt((2*3.1415928)**n * LA.det(var)) * exp(-1/2 * sum((X * LA.inv(Sigma) * X.T), 2))
    except ZeroDivisionError:
        pdf = 0
    return pdf


def main():
    X, Y = get_data()
    maax = 0
    Tr, TrL = X[0:280,:], Y[0:280]
    Te, TeL = X[280:,:], Y[280:]
    for i in range(50):
        gmix = gmm.GaussianMixture(n_components=4, covariance_type='full')
        pdf = expected(Tr)
        gmix.fit(Tr,TrL)
        result = gmix.predict(Te)
        error = TeL - result
        corr = float(np.count_nonzero(error==0)) / float(len(error))
        maax += corr

    
    print("Gmm accuracy:",maax * 10 / 3,"%")


if __name__=="__main__":
    main()

