'''
    CIFAR-10 Image Classification using Keras
'''


import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
np.random.seed(123)


# importing training and test data
(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# preprocessing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# model architecture
model = Sequential()

model.add(Convolution2D(16,2,2, activation= 'relu', input_shape = (32,32,3)))
model.add(Convolution2D(16,2,2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))
# loss function and optimization
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# training 
model.fit(x_train, y_train, batch_size=100, nb_epoch = 100)

score = model.evaluate(x_test, y_test)

print("\n\nscore: ", score)

input("\n\nexit?")