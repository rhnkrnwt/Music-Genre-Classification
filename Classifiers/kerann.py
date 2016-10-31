import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from total_arrange import get_data

A1, Y, Ate, Yte = get_data('dataset3.csv', 4, 'hot')

# print(A1[A1[:, 0] < -30])

np.set_printoptions(threshold=np.nan)
A1 = preprocessing.scale(A1)
Ate = preprocessing.scale(Ate)
np.random.seed(7)

model = Sequential()
model.add(Dense(30, input_dim=135, init='uniform', activation='relu'))
for i in range(9):
    model.add(Dense(30, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(A1, Y, nb_epoch=30, batch_size=1)

print("")
print("")

scores = model.evaluate(A1, Y)
print("")
print("Overall Training Accuracy : {0}%".format(scores[1]*100))


predictions = model.predict(Ate)
scores = model.evaluate(Ate, Yte)
print("")
print("Overall Test Accuracy : {0}%".format(scores[1]*100))
