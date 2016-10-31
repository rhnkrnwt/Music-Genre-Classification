import numpy as np
from total_arrange import get_data
import pickle
from decision_tree import make_tree, classify_one


def gen_tree(A, n):
    for i in range(n):
        print(A.shape[0])
        As = A[np.random.choice(A.shape[0], 200, replace=False), :]
        print(As.shape)

        tree = make_tree(As)
        with open('./forest/d_tree' + str(20 + i) + '.pickle', 'wb') as f:
            pickle.dump(tree, f)

def combine(Ate, Yte, num_trees, num_class):
    tree = [0] * num_trees
    for i in range(num_trees):
        with open('./forest/d_tree' + str(i) + '.pickle', 'rb') as f:
            tree[i] = pickle.load(f)

    pred = [0] * num_trees
    count_class = [0] * num_class
    tru = 0
    for i in range(Ate.shape[0]):
        count_class = [0] * num_class
        for j in range(num_trees):
            pred[j] = classify_one(Ate[i, :], tree[j])
            # print(pred[j])
            for key, _ in pred[j].items():
                # print(key)
                count_class[int(key)] += 1
        pr = count_class.index(max(count_class))
        if pr == Yte[i]:
            tru += 1

    print(tru / Ate.shape[0])



def rf():
    A, Y, Ate, Yte = get_data('dataset4.csv', num=5)
    Y = Y.reshape(-1, 1)
    A = np.concatenate((A, Y), axis=1)
    gen_tree(A, 200)
    # combine(Ate, Yte, 20, 5)

rf()
