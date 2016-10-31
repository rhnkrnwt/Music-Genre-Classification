import csv
import numpy as np

def num_to_lis(n):
    if n == 0:
        return [0, 0, 0, 1]
    if n == 1:
        return [0, 0, 0, 0]
    if n == 2:
        return [1, 1, 1, 1]
    if n == 3:
        return [1, 0, 0, 0]
    if n == 4:
        return [1, 1, 0, 0]

def get_data(fname, num=5, output_format='num'):
    with open('../Data/' + fname, 'r') as f:
        lines = csv.reader(f)
        F = list(lines)

    for i in F:
        for k in range(len(i)):
            try:
                i[k] = float(i[k])
            except:
                pass

    D = [0] * num
    for i in range(num):
        D[i] = np.array(F[100*i : 100*(i+1)])

    A = np.array([B[0, :] for B in D])
    Y = None
    if output_format == 'num':
        Y = np.array([x for x in range(num)])
    else:
        Y = np.array([num_to_lis(x) for x in range(num)])

    for i in range(1, 70):
        A = np.append(A, [B[i, :] for B in D], axis=0)
        if output_format == 'num':
            Y = np.append(Y, [x for x in range(num)], axis=0)
        else:
            Y = np.append(Y, [num_to_lis(x) for x in range(num)], axis=0)

    Ate = np.array([B[70, :] for B in D])
    Yte = None
    if output_format == 'num':
        Yte = np.array([x for x in range(num)])
    else:
        Yte = np.array([num_to_lis(x) for x in range(num)])

    for i in range(71, 100):
        Ate = np.append(Ate, [B[i, :] for B in D], axis=0)
        if output_format == 'num':
            Yte = np.append(Yte, [x for x in range(num)], axis=0)
        else:
            Yte = np.append(Yte, [num_to_lis(x) for x in range(num)], axis=0)

    return A, Y, Ate, Yte

get_data('dataset3.csv', output_format='hot')
