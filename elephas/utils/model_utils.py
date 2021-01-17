from enum import Enum

import tensorflow
import numpy as np
from pyspark import RDD


class ModelType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


class LossModelTypeMapper(Singleton):
    """
    Mapper for losses -> model type
    """

    def __init__(self):
        loss_to_model_type = {}
        loss_to_model_type.update(
            {'mean_squared_error': ModelType.REGRESSION,
             'mean_absolute_error': ModelType.REGRESSION,
             'mse': ModelType.REGRESSION,
             'mae': ModelType.REGRESSION,
             'cosine_proximity': ModelType.REGRESSION,
             'mean_absolute_percentage_error': ModelType.REGRESSION,
             'mean_squared_logarithmic_error': ModelType.REGRESSION,
             'logcosh': ModelType.REGRESSION,
             'binary_crossentropy': ModelType.CLASSIFICATION,
             'categorical_crossentropy': ModelType.CLASSIFICATION,
             'sparse_categorical_crossentropy': ModelType.CLASSIFICATION})
        self.__mapping = loss_to_model_type

    def get_model_type(self, loss):
        return self.__mapping.get(loss)

    def register_loss(self, loss, model_type):
        if callable(loss):
            loss = loss.__name__
        self.__mapping.update({loss: model_type})


def compute_predictions(model: tensorflow.keras.models.Model, model_type: ModelType, rdd: RDD, features: np.array):
    predict_function = determine_predict_function(model, model_type)
    predictions = rdd.ctx.parallelize(predict_function(features)).coalesce(1)
    if model_type == ModelType.CLASSIFICATION:
        predictions = predictions.map(lambda x: tuple(str(x)))
    else:
        predictions = predictions.map(lambda x: tuple([float(x)]))
    results_rdd = rdd.zip(predictions).map(lambda x: x[0] + x[1])
    return results_rdd


def determine_predict_function(model: tensorflow.keras.models.Model,
                               model_type: ModelType):
    if model_type == ModelType.CLASSIFICATION:
        if isinstance(model, tensorflow.keras.models.Sequential):
            predict_function = model.predict_classes
        else:
            # support for functional API
            predict_function = lambda x: model.predict(x).argmax(axis=-1)
    else:
        predict_function = model.predict

    return predict_function
