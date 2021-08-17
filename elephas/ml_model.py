import copy
import json
import warnings
from functools import partial

import h5py
import numpy as np
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasOutputCol, HasFeaturesCol, HasLabelCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StructField, ArrayType
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import get as get_optimizer

from .mllib import from_vector
from .spark_model import SparkModel
from .utils.model_utils import LossModelTypeMapper, ModelType, ModelTypeEncoder, as_enum
from .ml.adapter import df_to_simple_rdd
from .ml.params import *


class ElephasEstimator(Estimator, HasCategoricalLabels, HasValidationSplit, HasKerasModelConfig, HasFeaturesCol,
                       HasLabelCol, HasMode, HasEpochs, HasBatchSize, HasFrequency, HasVerbosity, HasNumberOfClasses,
                       HasNumberOfWorkers, HasOutputCol, HasLoss,
                       HasMetrics, HasKerasOptimizerConfig, HasCustomObjects, DefaultParamsReadable,
                       DefaultParamsWritable):
    """
    SparkML Estimator implementation of an elephas model. This estimator takes all relevant arguments for model
    compilation and training.

    Returns a trained model in form of a SparkML Model, which is also a Transformer.
    """

    @keyword_only
    def __init__(self, **kwargs):
        super(ElephasEstimator, self).__init__()
        # provide default for output column, if one is not supplied using `set_` method
        self._defaultParamMap[self.outputCol] = 'prediction'
        self.set_params(**kwargs)

    def get_config(self):
        return {'keras_model_config': self.get_keras_model_config(),
                'mode': self.get_mode(),
                'frequency': self.get_frequency(),
                'num_workers': self.get_num_workers(),
                'categorical': self.get_categorical_labels(),
                'loss': self.get_loss(),
                'metrics': self.get_metrics(),
                'validation_split': self.get_validation_split(),
                'featuresCol': self.getFeaturesCol(),
                'labelCol': self.getLabelCol(),
                'epochs': self.get_epochs(),
                'batch_size': self.get_batch_size(),
                'verbose': self.get_verbosity(),
                'nb_classes': self.get_nb_classes(),
                'outputCol': self.getOutputCol()}

    def save(self, file_name: str):
        f = h5py.File(file_name, mode='w')

        f.attrs['distributed_config'] = json.dumps({
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }).encode('utf8')

        f.flush()
        f.close()

    @keyword_only
    def set_params(self, **kwargs):
        """Set all provided parameters, otherwise set defaults
        """
        return self._set(**kwargs)

    def get_model(self):
        return model_from_json(self.get_keras_model_config(), self.get_custom_objects())

    def _fit(self, df: DataFrame):
        """Private fit method of the Estimator, which trains the model.
        """
        simple_rdd = df_to_simple_rdd(df, categorical=self.get_categorical_labels(), nb_classes=self.get_nb_classes(),
                                      features_col=self.getFeaturesCol(), label_col=self.getLabelCol())
        simple_rdd = simple_rdd.repartition(self.get_num_workers())
        keras_model = model_from_json(self.get_keras_model_config(), self.get_custom_objects())
        metrics = self.get_metrics()
        loss = self.get_loss()
        optimizer = get_optimizer(self.get_optimizer_config())
        keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        spark_model = SparkModel(model=keras_model,
                                 mode=self.get_mode(),
                                 frequency=self.get_frequency(),
                                 num_workers=self.get_num_workers(),
                                 custom_objects=self.get_custom_objects())
        spark_model.fit(simple_rdd,
                        epochs=self.get_epochs(),
                        batch_size=self.get_batch_size(),
                        verbose=self.get_verbosity(),
                        validation_split=self.get_validation_split())

        model_weights = spark_model.master_network.get_weights()
        return ElephasTransformer(labelCol=self.getLabelCol(),
                                  outputCol=self.getOutputCol(),
                                  featuresCol=self.getFeaturesCol(),
                                  keras_model_config=spark_model.master_network.to_json(),
                                  weights=model_weights,
                                  custom_objects=self.get_custom_objects(),
                                  model_type=LossModelTypeMapper().get_model_type(loss),
                                  history=spark_model.training_histories)

    def setFeaturesCol(self, value):
        warnings.warn("setFeaturesCol is deprecated in Spark 3.0.x+ - please supply featuresCol in the constructor i.e;"
                      " ElephasEstimator(featuresCol='foo')", DeprecationWarning)
        return self._set(featuresCol=value)

    def setLabelCol(self, value):
        warnings.warn("setLabelCol is deprecated in Spark 3.0.x+ - please supply labelCol in the constructor i.e;"
                      " ElephasEstimator(labelCol='foo')", DeprecationWarning)
        return self._set(labelCol=value)

    def setOutputCol(self, value):
        warnings.warn("setOutputCol is deprecated in Spark 3.0.x+ - please supply outputCol in the constructor i.e;"
                      " ElephasEstimator(outputCol='foo')", DeprecationWarning)
        return self._set(outputCol=value)

def load_ml_estimator(file_name: str) -> ElephasEstimator:
    f = h5py.File(file_name, mode='r')
    elephas_conf = json.loads(f.attrs.get('distributed_config'))
    config = elephas_conf.get('config')
    return ElephasEstimator(**config)


class ElephasTransformer(Model, HasKerasModelConfig, HasLabelCol, HasOutputCol, HasFeaturesCol, HasCustomObjects,
                         HasInferenceBatchSize):
    """SparkML Transformer implementation. Contains a trained model,
    with which new feature data can be transformed into labels.
    """

    @keyword_only
    def __init__(self, **kwargs):
        super(ElephasTransformer, self).__init__()
        if "weights" in kwargs.keys():
            # Strip model weights from parameters to init Transformer
            self.weights = kwargs.pop('weights')
        if "model_type" in kwargs.keys():
            # Extract loss from parameters
            self.model_type = kwargs.pop('model_type')
        self._history = kwargs.pop("history", [])
        self.set_params(**kwargs)

    @property
    def history(self):
        return self._history

    @keyword_only
    def set_params(self, **kwargs):
        """Set all provided parameters, otherwise set defaults
        """
        return self._set(**kwargs)

    def get_config(self):
        return {'keras_model_config': self.get_keras_model_config(),
                'labelCol': self.getLabelCol(),
                'featuresCol': self.getFeaturesCol(),
                'outputCol': self.getOutputCol(),
                'weights': [weight.tolist() for weight in getattr(self, 'weights', [])],
                'model_type': getattr(self, 'model_type', None)}

    def save(self, file_name: str):
        f = h5py.File(file_name, mode='w')

        f.attrs['distributed_config'] = json.dumps({
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }, cls=ModelTypeEncoder).encode('utf8')

        f.flush()
        f.close()

    def get_model(self):
        return model_from_json(self.get_keras_model_config(), self.get_custom_objects())

    def _transform(self, df):
        """Private transform method of a Transformer. This serves as batch-prediction method for our purposes.
        """
        output_col = self.getOutputCol()
        new_schema = copy.deepcopy(df.schema)
        rdd = df.rdd
        weights = rdd.ctx.broadcast(self.weights)

        def batched_prediction(data, inference_batch_size, features_col, predict_function) -> np.ndarray:
            # Do prediction in batches instead of materializing the whole partition at once
            batch = []
            preds = []
            for row in data:
                if len(batch) < inference_batch_size:
                    batch.append(from_vector(row[features_col]))
                else:
                    batch_np = np.array(batch)
                    pred = predict_function(batch_np)
                    preds.append(pred)
                    batch = [from_vector(row[features_col])]
            if len(batch) > 0:
                batch_np = np.array(batch)
                pred = predict_function(batch_np)
                preds.append(pred)

            if preds:
                res = np.vstack(preds)
                return res
            else:
                return np.array([])

        def extract_features_and_predict(model_yaml: str,
                                         custom_objects: dict,
                                         features_col: str,
                                         data,
                                         inference_batch_size: int = None):
            model = model_from_json(model_yaml, custom_objects)
            model.set_weights(weights.value)
            if inference_batch_size is not None and inference_batch_size > 0:
                return batched_prediction(data, inference_batch_size, features_col, model.predict)
            else:
                return model.predict(np.array([from_vector(x[features_col]) for x in data]))

        predictions = rdd.mapPartitions(
            partial(extract_features_and_predict,
                    self.get_keras_model_config(),
                    self.get_custom_objects(),
                    self.getFeaturesCol(),
                    inference_batch_size=self.get_inference_batch_size()
                    )
        )
        if self.model_type == ModelType.REGRESSION:
            predictions = predictions.map(lambda x: tuple([float(x)]))
            output_col_field = StructField(output_col, DoubleType(), True)
        else:
            # we're doing classification and predicting class probabilities
            predictions = predictions.map(lambda x: tuple([x.tolist()]))
            output_col_field = StructField(output_col, ArrayType(DoubleType()), True)
        results_rdd = rdd.zip(predictions).map(lambda x: x[0] + x[1])

        new_schema.add(output_col_field)
        results_df = df.sql_ctx.createDataFrame(results_rdd, new_schema)

        return results_df


def load_ml_transformer(file_name: str):
    f = h5py.File(file_name, mode='r')
    elephas_conf = json.loads(f.attrs.get('distributed_config'), object_hook=as_enum)
    config = elephas_conf.get('config')
    config['weights'] = [np.array(weight) for weight in config['weights']]
    return ElephasTransformer(**config)
