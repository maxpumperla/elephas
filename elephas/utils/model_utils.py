import json
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


def determine_predict_function(model: tensorflow.keras.models.Model,
                               model_type: ModelType,
                               predict_classes: bool = True):
    if model_type == ModelType.CLASSIFICATION and predict_classes:
        if isinstance(model, tensorflow.keras.models.Sequential):
            predict_function = model.predict_classes
        else:
            # support for functional API
            predict_function = lambda x: model.predict(x).argmax(axis=-1)
    else:
        predict_function = model.predict

    return predict_function


class ModelTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj in [e for e in ModelType]:
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(ModelType, member)
    else:
        return d
