import random
from math import isclose

from tensorflow.keras.optimizers import SGD

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

import pytest
import numpy as np

# enumerate possible combinations for training mode and parameter server for a classification model while also validatiing
# multiple workers for repartitioning
@pytest.mark.parametrize('mode,parameter_server_mode,num_workers',
                         [('synchronous', None, None),
                          ('synchronous', None, 2),
                          ('asynchronous', 'http', None),
                          ('asynchronous', 'http', 2),
                          ('asynchronous', 'socket', None),
                          ('asynchronous', 'socket', 2),
                          ('hogwild', 'http', None),
                          ('hogwild', 'http', 2),
                          ('hogwild', 'socket', None),
                          ('hogwild', 'socket', 2)])
def test_training_classification(spark_context, mode, parameter_server_mode, num_workers, mnist_data, classification_model):
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

    # Initialize SparkModel from keras model and Spark context
    spark_model = SparkModel(classification_model, frequency='epoch', num_workers=num_workers,
                             mode=mode, parameter_server_mode=parameter_server_mode, port=4000 + random.randint(0, 800))

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    # run inference on trained spark model
    predictions = spark_model.predict(x_test)
    # run evaluation on trained spark model
    evals = spark_model.evaluate(x_test, y_test)

    # assert we can supply rdd and get same prediction results when supplying numpy array
    test_rdd = spark_context.parallelize(x_test)
    assert [np.argmax(x) for x in predictions] == [np.argmax(x) for x in spark_model.predict(test_rdd)]

    # assert we get the same prediction result with calling predict on keras model directly
    assert [np.argmax(x) for x in predictions] == [np.argmax(x) for x in spark_model.master_network.predict(x_test)]

    # assert we get the same evaluation results when calling evaluate on keras model directly
    assert isclose(evals[0], spark_model.master_network.evaluate(x_test, y_test)[0], abs_tol=0.01)
    assert isclose(evals[1], spark_model.master_network.evaluate(x_test, y_test)[1], abs_tol=0.01)


# enumerate possible combinations for training mode and parameter server for a regression model while also validating
# multiple workers for repartitioning
@pytest.mark.parametrize('mode,parameter_server_mode,num_workers',
                         [('synchronous', None, None),
                          ('synchronous', None, 2),
                          ('asynchronous', 'http', None),
                          ('asynchronous', 'http', 2),
                          ('asynchronous', 'socket', None),
                          ('asynchronous', 'socket', 2),
                          ('hogwild', 'http', None),
                          ('hogwild', 'http', 2),
                          ('hogwild', 'socket', None),
                          ('hogwild', 'socket', 2)])
def test_training_regression(spark_context, mode, parameter_server_mode, num_workers, boston_housing_dataset,
                             regression_model):
    x_train, y_train, x_test, y_test = boston_housing_dataset
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Define basic parameters
    batch_size = 64
    epochs = 10
    sgd = SGD(lr=0.0000001)
    regression_model.compile(sgd, 'mse', ['mae', 'mean_absolute_percentage_error'])
    spark_model = SparkModel(regression_model, frequency='epoch', mode=mode, num_workers=num_workers,
                             parameter_server_mode=parameter_server_mode, port=4000 + random.randint(0, 800))

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    # run inference on trained spark model
    predictions = spark_model.predict(x_test)
    # run evaluation on trained spark model
    evals = spark_model.evaluate(x_test, y_test)

    # assert we can supply rdd and get same prediction results when supplying numpy array
    test_rdd = spark_context.parallelize(x_test)
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.predict(test_rdd)))

    # assert we get the same prediction result with calling predict on keras model directly
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.master_network.predict(x_test)))

    # assert we get the same evaluation results when calling evaluate on keras model directly
    assert isclose(evals[0], spark_model.master_network.evaluate(x_test, y_test)[0], abs_tol=0.01)
    assert isclose(evals[1], spark_model.master_network.evaluate(x_test, y_test)[1], abs_tol=0.01)
    assert isclose(evals[2], spark_model.master_network.evaluate(x_test, y_test)[2], abs_tol=0.01)

