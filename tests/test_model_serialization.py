import pytest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

from elephas.spark_model import SparkModel


def test_sequential_serialization(spark_context, classification_model):
    classification_model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["acc"])
    spark_model = SparkModel(classification_model, frequency='epoch', mode='synchronous')
    spark_model.save("elephas_sequential.h5")


def test_model_serialization():
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

    spark_model = SparkModel(model, frequency='epoch',
                             mode='synchronous', foo="bar")
    spark_model.save("elephas_model.h5")

