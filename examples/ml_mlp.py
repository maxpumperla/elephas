from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from elephas.ml_model import ElephasEstimator, ElephasTransformer
from elephas.ml.adapter import to_data_frame

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
df = to_data_frame(sc, X_train, Y_train, categorical=True)
test_df = to_data_frame(sc, X_test, Y_test, categorical=True)

# Initialize Spark ML Estimator
estimator = ElephasEstimator(sc,model, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1, num_workers=8, categorical=True, nb_classes=nb_classes)

# Fitting a model returns a Transformer
fitted_model = estimator.fit(df)

# Evaluate Spark model by evaluating the underlying model
prediction = fitted_model.transform(df)
pnl = prediction.select("label", "prediction")
pnl.show(30)
