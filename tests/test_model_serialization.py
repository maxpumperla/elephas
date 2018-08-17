from __future__ import absolute_import
from __future__ import print_function
import pytest

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.utils import np_utils

from elephas.spark_model import SparkModel, load_spark_model
from elephas.utils.rdd_utils import to_simple_rdd


# Define basic parameters
batch_size = 64
nb_classes = 10
epochs = 1

# Create Spark context
pytest.mark.usefixtures("spark_context")


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["acc"])


def test_sequential_serialization():
    spark_model = SparkModel(model, frequency='epoch', mode='synchronous')
    spark_model.save("elephas_sequential.h5")


# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def test_model_serialization():
    spark_model = SparkModel(model, frequency='epoch', mode='synchronous')
    spark_model.save("elephas_model.h5")
