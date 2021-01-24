import warnings
import numpy as np
import copy
import h5py
import json

from pyspark.ml.param.shared import HasOutputCol, HasFeaturesCol, HasLabelCol
from pyspark import keyword_only, RDD
from pyspark.ml import Estimator, Model
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, DoubleType, StructField

from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import get as get_optimizer

from .spark_model import SparkModel
from .utils.model_utils import LossModelTypeMapper, compute_predictions
from .utils.rdd_utils import from_vector
from .ml.adapter import df_to_simple_rdd
from .ml.params import *


class ElephasEstimator(Estimator, HasCategoricalLabels, HasValidationSplit, HasKerasModelConfig, HasFeaturesCol,
                       HasLabelCol, HasMode, HasEpochs, HasBatchSize, HasFrequency, HasVerbosity, HasNumberOfClasses,
                       HasNumberOfWorkers, HasOutputCol, HasLoss,
                       HasMetrics, HasKerasOptimizerConfig, HasCustomObjects):
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
        return model_from_yaml(self.get_keras_model_config(), self.get_custom_objects())

    def _fit(self, df: DataFrame):
        """Private fit method of the Estimator, which trains the model.
        """
        simple_rdd = df_to_simple_rdd(df, categorical=self.get_categorical_labels(), nb_classes=self.get_nb_classes(),
                                      features_col=self.getFeaturesCol(), label_col=self.getLabelCol())
        simple_rdd = simple_rdd.repartition(self.get_num_workers())
        keras_model = model_from_yaml(self.get_keras_model_config(), self.get_custom_objects())
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
        weights = simple_rdd.ctx.broadcast(model_weights)
        return ElephasTransformer(labelCol=self.getLabelCol(),
                                  outputCol=self.getOutputCol(),
                                  featuresCol=self.getFeaturesCol(),
                                  keras_model_config=spark_model.master_network.to_yaml(),
                                  weights=weights,
                                  custom_objects=self.get_custom_objects(),
                                  loss=loss)

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


def load_ml_estimator(file_name):
    f = h5py.File(file_name, mode='r')
    elephas_conf = json.loads(f.attrs.get('distributed_config'))
    config = elephas_conf.get('config')
    return ElephasEstimator(**config)


class ElephasTransformer(Model, HasKerasModelConfig, HasLabelCol, HasOutputCol, HasFeaturesCol, HasCustomObjects):
    """SparkML Transformer implementation. Contains a trained model,
    with which new feature data can be transformed into labels.
    """

    @keyword_only
    def __init__(self, **kwargs):
        super(ElephasTransformer, self).__init__()
        if "weights" in kwargs.keys():
            # Strip model weights from parameters to init Transformer
            self.weights = kwargs.pop('weights')
        if "loss" in kwargs.keys():
            # Extract loss from parameters
            self.model_type = LossModelTypeMapper().get_model_type(kwargs.pop('loss'))
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, **kwargs):
        """Set all provided parameters, otherwise set defaults
        """
        return self._set(**kwargs)

    def get_config(self):
        return {'keras_model_config': self.get_keras_model_config(),
                'labelCol': self.getLabelCol(),
                'outputCol': self.getOutputCol()}

    def save(self, file_name: str):
        f = h5py.File(file_name, mode='w')

        f.attrs['distributed_config'] = json.dumps({
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }).encode('utf8')

        f.flush()
        f.close()

    def get_model(self):
        return model_from_yaml(self.get_keras_model_config())

    def _transform(self, df):
        """Private transform method of a Transformer. This serves as batch-prediction method for our purposes.
        """
        output_col = self.getOutputCol()
        label_col = self.getLabelCol()
        features_col = self.getFeaturesCol()
        new_schema = copy.deepcopy(df.schema)
        new_schema.add(StructField(output_col, StringType(), True))

        rdd = df.rdd.coalesce(1)
        features = np.asarray(rdd.map(lambda x: from_vector(x[features_col])).collect())
        # Note that we collect, since executing this on the rdd would require model serialization once again
        model = model_from_yaml(self.get_keras_model_config(), self.get_custom_objects())
        model.set_weights(self.weights.value)

        results_rdd = compute_predictions(model, self.model_type, rdd, features)
        results_df = df.sql_ctx.createDataFrame(results_rdd, new_schema)
        results_df = results_df.withColumn(
            output_col, results_df[output_col].cast(DoubleType()))
        results_df = results_df.withColumn(
            label_col, results_df[label_col].cast(DoubleType()))

        return results_df


def load_ml_transformer(file_name: str):
    f = h5py.File(file_name, mode='r')
    elephas_conf = json.loads(f.attrs.get('distributed_config'))
    config = elephas_conf.get('config')
    return ElephasTransformer(**config)