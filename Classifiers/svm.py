import numpy as np
from sklearn import svm, preprocessing
from svm_arrange import get_data

A1, Y, Ate, Yte = get_data()
clf = svm.NuSVC(kernel='rbf', nu=0.2)

A1 = preprocessing.scale(A1)
Ate = preprocessing.scale(Ate)

clf.fit(A1, Y)
print(clf.predict(Ate))
prediction = clf.predict(Ate)

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
