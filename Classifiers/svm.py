import numpy as np
from sklearn import svm, preprocessing
from svm_arrange import get_data

A1, Y, Ate, Yte = get_data()
clf = svm.NuSVC()

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

print(mis)
