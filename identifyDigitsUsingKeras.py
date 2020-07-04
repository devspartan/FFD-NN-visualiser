from __future__ import print_function
import numpy as np
import pandas as pd
import datetime
from keras.callbacks.tensorboard_v2 import TensorBoard
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model, print_summary
import graphviz
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model

# data: shuffled and split between train and test sets
xTrain = np.array(pd.read_csv('mnist_train.csv').iloc[:, 1:])
yTrain = np.array(pd.read_csv('mnist_train.csv').iloc[:, 0])
xTest = np.array(pd.read_csv('mnist_test.csv').iloc[:, 1:])
yTest = np.array(pd.read_csv('mnist_test.csv').iloc[:, 0])


# normalize training data between 0 and 1
xTrain = xTrain / 255
xTest = xTest / 255
print(xTrain.shape[0], 'train samples')
print(xTest.shape[0], 'test samples')

# convert class vectors to binary class matrices
yTrain = to_categorical(yTrain, 10)
yTest = to_categorical(yTest, 10)


# for tensorflow Board  need learning
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# define model
model = Sequential()
model.add(Dense(100, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
print('----------------------------')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(xTrain, yTrain, batch_size=64, epochs=10, verbose=2, validation_data=(xTest, yTest))
#
# # plotting accuracy vs epoch
# plt.plot(history.history['val_accuracy'])
# plt.plot(history.history['accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.grid(color='#8c8c8c', linestyle='-', linewidth=0.5)
# plt.show()
#
# # plottting loss vs epoch
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.grid(color='#8c8c8c', linestyle='-', linewidth=0.5)
# plt.show()


# load preSaved local Keras model
model = load_model('kerasCNNmodel.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# print(model.optimizer)
# print(model.get_weights())


# evaluation of model
score = model.evaluate(xTest, yTest, verbose=2)
print("Test score:", score[0])
print('Test accuracy:', score[1])


# predict using model
index = 1233
a = xTrain[index, :]
a = np.reshape(a, [1, 784])
ans = model.predict(a)                     #returns a numpy array of predictions
print('Probability:', ans.max())
print('Prediction:', ans.argmax())
print('Actual:', yTrain[index, ].argmax())


# -------------- saving model -------------------

# model.save('kerasModel.h5')                 #saves architecture and weight of the model
# plot_model(model, to_file='model.png')       #saving model architect as png



# saving only architecture to jason
# kerasJason = model.to_json()

# saving only weights
# model.save_weights('myKerasWeight.h5')

model2 = Sequential([Dense(100, activation='relu', input_shape=(784, )),
                     Dense(100, activation='relu'),
                     Dense(10, activation='softmax')])
model2.load_weights('myKerasWeight.h5')              #load preTrained model weights

# preduct using model2
index = 1212
a = xTrain[index, :]
a = np.reshape(a, [1, 784])
ans = model2.predict(a)                   #returns a numpy array of predictions
print('Probability:', ans.max())
print('Prediction:', ans.argmax())
print('Actual:', yTrain[index, ].argmax())

print('---------------------------')