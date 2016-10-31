import csv
import numpy as np

def get_data():
    num = 4
    with open('../Data/dataset4.csv', 'r') as f:
        lines = csv.reader(f)
        F = list(lines)

    for i in F:
        for k in range(len(i)):
            try:
                i[k] = float(i[k])
            except:
                pass

    bias = np.ones((280, 1))
    F0 = np.array(F[0:100])
    F1 = np.array(F[100:200])
    F2 = np.array(F[200:300])
    F3 = np.array(F[300:400])
    # print(F0.shape, F1.shape, F2.shape, F3.shape)
    A = np.array([F0[0, :], F1[0, :], F2[0, :], F3[0, :]])
    Y = np.array([0, 1, 2, 3])

    for i in range(1, 70):
        A = np.append(A, [F0[i, :], F1[i, :], F2[i, :], F3[i, :]], axis=0)
        Y = np.append(Y, [0, 1, 2, 3], axis=0)
    # print(A.shape)
    # print(bias.shape)
    # A = np.concatenate((A, bias), axis=1)
    # print(A.shape)

    Ate_pop = np.array([F0[70, :]])
    Yte_pop = np.array([0])

    Ate_jazz = np.array([F1[70, :]])
    Yte_jazz = np.array([1])

    Ate_metal = np.array([F2[70, :]])
    Yte_metal = np.array([2])

    Ate_classical = np.array([F3[70, :]])
    Yte_classical = np.array([3])

    Ate = np.array([F0[70, :], F1[70, :], F2[70, :], F3[70, :]])
    Yte = np.array([0, 1, 2, 3])

    for i in range(71, 100):
        Ate = np.append(Ate, [F0[i, :], F1[i, :], F2[i, :], F3[i, :]], axis=0)
        Yte = np.append(Yte, [0, 1, 2, 3], axis=0)

        Ate_pop = np.append(Ate_pop, [F0[i, :]], axis=0)
        Yte_pop = np.append(Yte_pop, [0], axis=0)

        Ate_jazz = np.append(Ate_jazz, [F1[i, :]], axis=0)
        Yte_jazz = np.append(Yte_jazz, [1], axis=0)

        Ate_metal = np.append(Ate_metal, [F2[i, :]], axis=0)
        Yte_metal = np.append(Yte_metal, [2], axis=0)

        Ate_classical = np.append(Ate_classical, [F3[i, :]], axis=0)
        Yte_classical = np.append(Yte_classical, [3], axis=0)

    # Yte = Yte.reshape(-1, 1)
    # Ate = np.concatenate((Ate, Yte), axis=1)
    # Ate = np.concatenate((Ate, bias), axis=1)

    return (A, Y, Ate, Yte, Ate_pop, Yte_pop, Ate_jazz, Yte_jazz, Ate_metal,
            Yte_metal, Ate_classical, Yte_classical)
