from __future__ import absolute_import
from __future__ import print_function

from keras.optimizers import SGD

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

import pytest


@pytest.mark.parametrize('mode', ['synchronous', 'asynchronous', 'hogwild'])
def test_training_modes_classification(spark_context, mode, mnist_data, classification_model):
    # Define basic parameters
    batch_size = 64
    epochs = 10

    # Load data
    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    sgd = SGD(lr=0.1)
    classification_model.compile(sgd, 'categorical_crossentropy', ['acc'])

    # Build RDD from numpy features and labels
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Initialize SparkModel from Keras model and Spark context
    spark_model = SparkModel(classification_model, frequency='epoch', mode=mode)

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)
    # Evaluate Spark model by evaluating the underlying model
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)

    # mnist is simple - our model should have started converging under each training mode after 10 epochs
    assert score[1] >= 0.7


@pytest.mark.parametrize('mode', ['synchronous', 'asynchronous', 'hogwild'])
def test_training_modes_regression(spark_context, mode, boston_housing_dataset, regression_model):
    x_train, y_train, x_test, y_test = boston_housing_dataset
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Define basic parameters
    batch_size = 64
    epochs = 10
    sgd = SGD(lr=0.0000001)
    regression_model.compile(sgd, 'mse', ['mae'])
    spark_model = SparkModel(regression_model, frequency='epoch', mode=mode)

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    assert score[1] <= 15
