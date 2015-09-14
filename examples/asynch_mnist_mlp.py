from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from elephas.asynch_spark_model import AsynchSparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf

# Define basic parameters
batch_size = 128
nb_classes = 10
nb_epoch = 20

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

# Compile model
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, Y_train)
#rdd = rdd.repartition(1)

# Initialize AsynchSparkModel from Keras model and Spark context
spark_model = AsynchSparkModel(sc,model)

# Train Spark model
print('Training model')
spark_model.train(rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1, num_workers=8)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.get_network().evaluate(X_test, Y_test, show_accuracy=True, verbose=2)
print('Test accuracy:', score[1])

