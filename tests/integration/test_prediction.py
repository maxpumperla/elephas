import random
from math import isclose

from tensorflow.keras.optimizers import SGD
import numpy as np

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

import pytest


@pytest.mark.parametrize('mode', ['synchronous', 'asynchronous', 'hogwild'])
def test_classification_prediction(spark_context, mode, mnist_data, classification_model):
    # Define basic parameters
    batch_size = 64
    epochs = 10

    # Load data
    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:100]
    y_test = y_test[:100]

    sgd = SGD(lr=0.1)
    classification_model.compile(sgd, 'categorical_crossentropy', ['acc'])

    # Build RDD from numpy features and labels
    train_rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Initialize SparkModel from keras model and Spark context
    spark_model = SparkModel(classification_model, frequency='epoch', mode=mode, port=4000 + random.randint(0, 200))

    # Train Spark model
    spark_model.fit(train_rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    # assert we have as many predictions as samples provided
    predictions = spark_model.predict(x_test)
    assert len(predictions) == 100

    test_rdd = spark_context.parallelize(x_test)
    # assert we can supply rdd and get same results
    assert [np.argmax(x) for x in predictions] == [np.argmax(x) for x in spark_model.predict(test_rdd)]
    # assert we get the same result with calling predict on keras model
    assert [np.argmax(x) for x in predictions] == [np.argmax(x) for x in spark_model.master_network.predict(x_test)]


@pytest.mark.parametrize('mode', ['synchronous', 'asynchronous', 'hogwild'])
def test_regression_prediction(spark_context, mode, boston_housing_dataset, regression_model):
    x_train, y_train, x_test, y_test = boston_housing_dataset
    train_rdd = to_simple_rdd(spark_context, x_train, y_train)
    x_test = x_test[:100]

    # Define basic parameters
    batch_size = 64
    epochs = 10
    sgd = SGD(lr=0.000001)
    regression_model.compile(sgd, 'mse', ['mae'])
    # Initialize SparkModel from keras model and Spark context
    spark_model = SparkModel(regression_model, frequency='epoch', mode=mode, port=4000 + random.randint(0, 200))

    # Train Spark model
    spark_model.fit(train_rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    # assert we have as many predictions as samples provided
    predictions = spark_model.predict(x_test)
    assert len(predictions) == 100

    test_rdd = spark_context.parallelize(x_test)
    # assert we can supply rdd and get same results
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.predict(test_rdd)))
    # assert we get the same result with calling predict on keras model
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.master_network.predict(x_test)))


@pytest.mark.parametrize('mode', ['synchronous', 'asynchronous', 'hogwild'])
def test_evaluate(spark_context, mode, mnist_data, classification_model):
    # Define basic parameters
    batch_size = 64
    epochs = 10

    # Load data
    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:100]
    y_test = y_test[:100]

    sgd = SGD(lr=0.1)
    classification_model.compile(sgd, 'categorical_crossentropy', ['acc'])

    # Build RDD from numpy features and labels
    train_rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Initialize SparkModel from keras model and Spark context
    spark_model = SparkModel(classification_model, frequency='epoch', mode=mode, port=4000 + random.randint(0, 200))

    # Train Spark model
    spark_model.fit(train_rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    results = spark_model.evaluate(x_test, y_test)

    assert isclose(results, spark_model.master_network.evaluate(x_test, y_test)[0], abs_tol=0.01)
