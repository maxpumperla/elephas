from tensorflow.keras.optimizers import SGD

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
    x_test = x_test[:1]
    y_test = y_test[:500]

    sgd = SGD(lr=0.1)
    classification_model.compile(sgd, 'categorical_crossentropy', ['acc'])

    # Build RDD from numpy features and labels
    train_rdd = to_simple_rdd(spark_context, x_train, y_train)
    test_rdd = spark_context.parallelize(x_test)

    # Initialize SparkModel from keras model and Spark context
    spark_model = SparkModel(classification_model, frequency='epoch', mode=mode)

    # Train Spark model
    spark_model.fit(train_rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    predictions = spark_model.predict(test_rdd)

