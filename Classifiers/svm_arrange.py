import csv
import numpy as np

def get_data():
    num = 4
    with open('../Data/dataset.csv', 'r') as f:
        lines = csv.reader(f)
        dataset = list(lines)


    D = []
    for i in range(0,len(dataset),15):
        tmp1 = dataset[i:i+15]
        tmp2 = [k for l in tmp1 for k in l]
        D.append(tmp2)

    F = [[]]*num
    for i in range(num):
        F[i] = D[i*100:(i+1)*100]
    for i in F:
        for j in i:
            for k in range(len(j)):
                try:
                    j[k] = float(j[k])
                except:
                    pass

    F0 = np.array(F[0])
    F1 = np.array(F[1])
    F2 = np.array(F[2])
    F3 = np.array(F[3])
    A = np.array([F0[0, :], F1[0, :], F2[0, :], F3[0, :]])
    Y = np.array([0, 1, 2, 3])
    for i in range(1, 70):
        A = np.append(A, [F0[i, :], F1[i, :], F2[i, :], F3[i, :]], axis=0)
        Y = np.append(Y, [0, 1, 2, 3], axis=0)

    Ate = np.array([F0[70, :], F1[70, :], F2[70, :], F3[70, :]])
    Yte = np.array([0, 1, 2, 3])

    for i in range(71, 100):
        Ate = np.append(Ate, [F0[i, :], F1[i, :], F2[i, :], F3[i, :]], axis=0)
        Yte = np.append(Yte, [0, 1, 2, 3], axis=0)

    return A, Y, Ate, Yte
