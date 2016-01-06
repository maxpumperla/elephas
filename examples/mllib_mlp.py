from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

from elephas.spark_model import SparkMLlibModel
from elephas.utils.rdd_utils import to_labeled_point, lp_to_simple_rdd
from elephas import optimizers as elephas_optimizers

from pyspark import SparkContext, SparkConf

# Define basic parameters
batch_size = 64
nb_classes = 10
nb_epoch = 3

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
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# Compile model
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
lp_rdd = to_labeled_point(sc, X_train, Y_train, categorical=True)
print(lp_rdd.first())
rdd = lp_to_simple_rdd(lp_rdd, True, nb_classes)
rdd = rdd.repartition(4)
rdd.first()

# Initialize SparkModel from Keras model and Spark context
adadelta = elephas_optimizers.Adadelta()
spark_model = SparkMLlibModel(sc,
                              model,
                              optimizer=adadelta,
                              frequency='batch',
                              mode='asynchronous',
                              num_workers=2)

# Train Spark model
spark_model.train(lp_rdd,
                  nb_epoch=20,
                  batch_size=32,
                  verbose=0,
                  validation_split=0.1,
                  categorical=True,
                  nb_classes=nb_classes)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.master_network().evaluate(X_test, Y_test,
                                           show_accuracy=True, verbose=2)
print('Test accuracy:', score[1])
