from keras.models import Sequential
from keras.layers import Dense
import numpy
import os
import theano
# theano.config.device = 'gpu'

seed = 7
numpy.random.seed(seed)

data_path = "/".join([os.getcwd(), "../Data/pima-indians-diabetes.data.txt"])
dataset = numpy.loadtxt(data_path, delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=150, batch_size=10)


scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))