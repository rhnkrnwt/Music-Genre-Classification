import numpy as np
from total_arrange import get_data
from math import log2
import decimal
import pickle

class dnode:
    def __init__(self, feature=-1, val=None, result=None, tb=None, fb=None):
        self.feature = feature
        self.val = val
        self.result = result
        self.tb = tb
        self.fb = fb

def divide(A, feature, val):
    A1 = A[A[:, feature] >= val]
    A2 = A[A[:, feature] < val]

    return (A1, A2)

def entropy(A):
    if A.shape[0] == 0:
        return 0
    ent = 0.0
    count = [0] * 5

    for i in range(A.shape[0]):
        idx = int(A[i, -1])
        count[idx] += 1
    p = 0.0
    for c in count:
        if c > 0:
            p = float(c) / float(A.shape[0])
            ent -= p * log2(p)

    return ent

def get_results(A):
    res = {}
    for r in A[:, -1]:
        if r not in res:
            res[r] = 0
        res[r] += 1
    return res

def make_tree(A, max_depth=None):

    if max_depth is not None and max_depth < 1:
        return dnode(result=get_results(A))
    if A.shape[0] == 0:
        return dnode()
    c_score = entropy(A)

    st = np.std(A, axis=0)
    inc = (st / A.shape[1])
    maxi = np.amax(A, axis=0)
    mini = np.amin(A, axis=0)

    max_gain = 0.0
    best_criteria = None
    best_divide = None

    for feat in range(A.shape[1] - 1):
        start = float(mini[feat])
        stop = float(maxi[feat])


        i = float(inc[feat])
        print(feat)
        if i == 0:
            continue
        slices = abs(int((stop - start) / i))

        for v in np.linspace(start, stop, slices, endpoint=False):
            sets = divide(A, feat, v)
            #print(sets[0].shape)
            #print(sets[1].shape)
            gain = 0.0
            p = 0.0
            if sets[0].shape[0] > 0 and sets[0].shape[0] > 0:
                p = float(sets[0].shape[0] / A.shape[0])
                gain = c_score - p * entropy(sets[0]) - (1 - p) * entropy(sets[1])

                if gain > max_gain:
                    max_gain = gain
                    best_criteria = (feat, v)
                    best_divide = sets

    if max_gain > 0:
        print("Best criteria: {0}".format(best_criteria))
        true_branch = None
        false_branch = None
        if max_depth is None:
            true_branch = make_tree(best_divide[0])
            false_branch = make_tree(best_divide[1])
        else:
            true_branch = make_tree(best_divide[0], max_depth - 1)
            false_branch = make_tree(best_divide[1], max_depth - 1)
        return dnode(feature=best_criteria[0], val=best_criteria[1],
                     tb=true_branch, fb=false_branch)
    return dnode(result=get_results(A))

def print_tree(tree, spacing=''):
   # Is this a leaf node?
    if tree.result != None:
        print(str(tree.result))
    else:
        print(str(tree.feature) + ':' + str(tree.val) + '? ')
        # Print the branches
        print(spacing + 'T->', end=" ")
        print_tree(tree.tb, spacing + '  ')
        print(spacing + 'F->', end=" ")
        print_tree(tree.fb, spacing + '  ')


def classify_one(sample, tree):
    if tree.result != None:
        return tree.result
    sample_val = sample[tree.feature]
    branch = None
    if sample_val >= tree.val:
        branch = tree.tb
    else:
        branch = tree.fb

    return (classify_one(sample, branch))


if __name__ == '__main__':
    A, Y, Ate, Yte = get_data('dataset4.csv', num=5)

    Y = Y.reshape(-1, 1)
    A = np.concatenate((A, Y), axis=1)

    # tree = make_tree(A, 3)
    # with open('d_tree.pickle', 'wb') as f:
    #     pickle.dump(tree, f)
    tree = None
    with open('forest/d_tree0.pickle', 'rb') as f:
        tree = pickle.load(f)

    print_tree(tree)
    tru = 0
    for i in range(Ate.shape[0]):
        pred = (classify_one(Ate[i, :], tree))
        print(pred, Yte[i])
        for key, _ in pred.items():
            print("{0} {1}".format(int(key), Yte[i]))
            if int(key) == Yte[i]:
                tru += 1
                print(key)
    print(tru / Ate.shape[0])

    # print(classify_one(Ate[0, :], tree))
    # print(Yte[0])
