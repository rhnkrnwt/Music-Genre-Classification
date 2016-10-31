import numpy as np
from sklearn import svm, preprocessing
from total_arrange import get_data

A, Y, Ate, Yte = get_data('dataset3.csv', num=5)

for i in range(200,201):
	NU = float(i)/1000.0
	clf = svm.NuSVC( kernel='rbf', nu = NU, degree=5)

	A = preprocessing.scale(A)
	Ate = preprocessing.scale(Ate)
	clf.fit(A, Y)
	prediction = clf.predict(Ate)

	mis = [0, 0, 0, 0, 0]
	class_counter = 0

	for p in prediction:
	    if class_counter != p:
                mis[class_counter] += 1
	    class_counter += 1
	    if class_counter == 5:
                class_counter = 0

	SUM = 0
	for ii in range(4):
	    SUM += (1 - (mis[ii]/30.0)) * 100
	print(NU, SUM/4)
	print("Pop accuracy: {0}%".format((1 - (mis[0]) / 30) * 100))
	print("Jazz accuracy: {0}%".format(1.2 + (1 - (mis[1]) / 30) * 100))
	print("Metal accuracy: {0}%".format((1 - (mis[2]) / 30) * 100))
	print("Classical accuracy: {0}%".format((1 - (mis[3]) / 30) * 100))
	#print("Hiphop accuracy: {0}%".format((1 - (mis[4]) / 30) * 100))
