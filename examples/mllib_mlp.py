# from __future__ import absolute_import
# from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K

from elephas.spark_model import SparkMLlibModel, SparkModel
from elephas.utils.rdd_utils import to_labeled_point, to_simple_rdd

from pyspark import SparkContext, SparkConf

# Define basic parameters
batch_size = 128
nb_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Flatten the data, MLP doesn't use the 2D structure of the data. 784 = 28*28
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#see http://cs231n.github.io/convolutional-networks/#overview
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# Make the value floats in [0;1] instead of int in [0;255]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
# Display the shapes to check if everything's ok
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices (ie one-hot vectors)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))

# model = Sequential()
# #For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
# #By default the stride/subsample is 1
# #border_mode "valid" means no zero-padding.
# #If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
#                         border_mode='valid',
#                         input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# #For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
# #Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes)) #Last layer with one output per class
# model.add(Activation('softmax')) #We want a score simlar to a probability for each class

model.summary()

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])

# #Make the model learn
# model.fit(x_train, y_train,
#           batch_size=batch_size, epochs=epochs,
#           verbose=2,
#           validation_data=(x_test, y_test))
#
# #Evaluate how the model does on the test set
# score = model.evaluate(x_test, y_test, verbose=0)
#
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP')
# .setMaster('local[8]')
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
# lp_rdd = to_labeled_point(sc, x_train, y_train, categorical=True)
rdd = to_simple_rdd(sc, x_train, y_train)

# Train Spark model
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

spark_model.fit(rdd, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_file='save/mlp.h5'
import os
if not os.path.exists("save/"):
    os.mkdir("save/")
model.save(model_file)
