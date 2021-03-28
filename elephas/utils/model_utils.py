import json
from enum import Enum
from typing import Union

from pyspark.sql import Column
import pyspark.sql.functions as F


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


def argmax(col: Union[str, Column]) -> Column:
    """
    returns expression for finding the argmax in an array column
    :param col: array column to find argmax of
    :return: expression which can be used in `select` or `withColumn`
    """
    return F.expr(f'array_position({col}, array_max({col})) - 1')
