import numpy as np
from sklearn import svm, preprocessing
from arrange_dt import get_data

A, Y, Ate, Yte, Ate_pop, Yte_pop, Ate_jazz, Yte_jazz, Ate_metal,\
Yte_metal, Ate_classical, Yte_classical, Ate_hiphop,\
Yte_hiphop = get_data()

clf = svm.NuSVC(kernel='rbf', nu=0.2)

A = preprocessing.scale(A)
Ate = preprocessing.scale(Ate)

clf.fit(A, Y)
print(clf.predict(Ate))
prediction = clf.predict(Ate)

class_counter = 0
mis = [0, 0, 0, 0, 0]

for p in prediction:
    if class_counter != p:
        mis[class_counter] += 1

    class_counter += 1
    if class_counter == 5:
        class_counter = 0

print("Pop accuracy: {0}%".format((1 - (mis[0]) / 30) * 100))
print("Jazz accuracy: {0}%".format((1 - (mis[1]) / 30) * 100))
print("Metal accuracy: {0}%".format((1 - (mis[2]) / 30) * 100))
print("Classical accuracy: {0}%".format((1 - (mis[3]) / 30) * 100))
print("Hiphop accuracy: {0}%".format((1 - (mis[4]) / 30) * 100))
