import pytest

from keras.models import Model
from keras.layers import Dense, Input

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


@pytest.mark.skip(reason="not feasible on travis right now")
def test_java_avg_serde():
    from elephas.dl4j import ParameterAveragingModel

    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    spark_model = ParameterAveragingModel(java_spark_context=None, model=model, num_workers=4, batch_size=32,
                                          averaging_frequency=5, num_batches_prefetch=0, collect_stats=False,
                                          save_file='temp.h5')
    spark_model.save("java_param_averaging_model.h5")


@pytest.mark.skip(reason="not feasible on travis right now")
def test_java_sharing_serde():
    from elephas.dl4j import ParameterSharingModel

    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    spark_model = ParameterSharingModel(java_spark_context=None, model=model, num_workers=4, batch_size=32,
                                        shake_frequency=0, min_threshold=1e-5, update_threshold=1e-3,
                                        workers_per_node=-1, num_batches_prefetch=0, step_delay=50, step_trigger=0.05,
                                        threshold_step=1e-5, collect_stats=False, save_file='temp.h5')
    spark_model.save("java_param_sharing_model.h5")

