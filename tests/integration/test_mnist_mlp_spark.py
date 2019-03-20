from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

import pytest
pytest.mark.usefixtures("spark_context")


def test_async_mode(spark_context):
    # Define basic parameters
    batch_size = 64
    nb_classes = 10
    epochs = 1

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

    sgd = SGD(lr=0.1)
    model.compile(sgd, 'categorical_crossentropy', ['acc'])

    # Build RDD from numpy features and labels
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Initialize SparkModel from Keras model and Spark context
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)
    # Evaluate Spark model by evaluating the underlying model
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    assert score[1] >= 0.7


if __name__ == '__main__':
    pytest.main([__file__])
