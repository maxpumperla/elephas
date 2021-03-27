import random

import numpy as np
import pytest
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from elephas.spark_model import SparkModel
from elephas.utils import to_simple_rdd


@pytest.mark.parametrize('mode', ['synchronous', 'asynchronous', 'hogwild'])
def test_training_custom_activation(mode, spark_context):
    def custom_activation(x):
        return sigmoid(x) + 1

    model = Sequential()
    model.add(Dense(1, input_dim=1, activation=custom_activation))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.1)
    model.compile(sgd, 'binary_crossentropy', ['acc'])

    x_train = np.random.rand(100)
    y_train = np.zeros(100)
    x_test = np.random.rand(10)
    y_test = np.zeros(10)
    y_train[:50] = 1
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    spark_model = SparkModel(model, frequency='epoch', mode=mode,
                             custom_objects={'custom_activation': custom_activation},
                             port=4000 + random.randint(0, 800))
    spark_model.fit(rdd, epochs=1, batch_size=16, verbose=0, validation_split=0.1)
    assert spark_model.predict(x_test)
    assert spark_model.evaluate(x_test, y_test)
