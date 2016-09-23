import numpy as np
from scipy.ndimage import zoom

def get_data():
    T0 = np.loadtxt("Dataset/data1")
    T3 = np.loadtxt("Dataset/data4")
    T7 = np.loadtxt("Dataset/data7")

    t0 = T0[0:32, :]
    t0 = zoom(t0, 1/4, order=0)
    t3 = T3[0:32, :]
    t3 = zoom(t3, 1/4, order=0)
    t7 = T7[0:32, :]
    t7 = zoom(t7, 1/4, order=0)

    # training set
    A1 = np.array([t0.flatten(), t3.flatten(), t7.flatten()])
    Y = np.array([[0, 1], [0, 0], [1, 1]])


    for i in range(148):
        t0 = T0[(i+1)*32:(i+2)*32, :]
        t0 = zoom(t0, 1/4, order=0)
        t3 = T3[(i+1)*32:(i+2)*32, :]
        t3 = zoom(t3, 1/4, order=0)
        t7 = T7[(i+1)*32:(i+2)*32, :]
        t7 = zoom(t7, 1/4, order=0)

        A1 = np.append(A1, [t0.flatten(), t3.flatten(),
                       t7.flatten()], axis=0)
        Y = np.append(Y, [[0, 1], [0, 0],  [1, 1]], axis=0)

    t0 = T0[150*32:151*32, :]
    t0 = zoom(t0, 1/4, order=0)
    t3 = T3[150*32:151*32, :]
    t3 = zoom(t3, 1/4, order=0)
    t7 = T7[150*32:151*32, :]
    t7 = zoom(t7, 1/4, order=0)

    # test set
    Ate = np.array([t0.flatten(), t3.flatten(), t7.flatten()])
    Yte = np.array([[0, 1], [0, 0], [1, 1]])
    yA = [[0, 1], [0, 0], [1, 1]]
    Tl = [T0, T3, T7]

    for i, l in enumerate(Tl):
        r = (l.shape[0] / 32) - 151

        for j in range(int(r)):
            t = l[(j+151)*32:(j+152)*32, :]
            t = zoom(t, 1/4, order=0)

            Ate = np.append(Ate, [t.flatten()], axis=0)
            Yte = np.append(Yte, [yA[i]], axis=0)

    return A1, Y, Ate, Yte
