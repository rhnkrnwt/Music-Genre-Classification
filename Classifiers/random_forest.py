import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from arrange_dt4 import get_data

def rf(md, A, Y, Ate, Yte):
    clf = RandomForestClassifier(max_depth=md, n_jobs=1, n_estimators=400)
    clf.fit(A, Y)

    return clf.predict(Ate)

if __name__ == '__main__':
    A, Y, Ate, Yte, Ate_pop, Yte_pop, Ate_jazz, Yte_jazz, Ate_metal,\
    Yte_metal, Ate_classical, Yte_classical = get_data()

    prediction = rf(int(sys.argv[1]), A, Y, Ate, Yte)

    class_counter = 0
    mis = [0, 0, 0, 0]
    for p in prediction:
        if class_counter != p:
            mis[class_counter] += 1

        class_counter += 1
        if class_counter == 4:
            class_counter = 0

    print("Pop accuracy: {0}%".format((1 - (mis[0]) / 30) * 100))
    print("Jazz accuracy: {0}%".format((1 - (mis[1]) / 30) * 100))
    print("Metal accuracy: {0}%".format((1 - (mis[2]) / 30) * 100))
    print("Classical accuracy: {0}%".format((1 - (mis[3]) / 30) * 100))
    #print("Hiphop accuracy: {0}%".format((1 - (mis[4]) / 30) * 100))
