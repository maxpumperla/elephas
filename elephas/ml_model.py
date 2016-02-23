from __future__ import absolute_import, print_function

import numpy as np

from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasFeaturesCol, HasLabelCol
from pyspark.ml.util import keyword_only
from pyspark.sql import Row
from pyspark.ml import Estimator, Model

from keras.models import model_from_yaml

from .spark_model import SparkModel
from .utils.rdd_utils import from_vector, to_vector
from .ml.adapter import df_to_simple_rdd
from .ml.params import *
from .optimizers import get


class ElephasEstimator(Estimator, HasCategoricalLabels, HasValidationSplit, HasKerasModelConfig, HasFeaturesCol, HasLabelCol, HasMode, HasEpochs, HasBatchSize,
                       HasFrequency, HasVerbosity, HasNumberOfClasses, HasNumberOfWorkers, HasOptimizerConfig):
    '''
    SparkML Estimator implementation of an elephas model. This estimator takes all relevant arguments for model
    compilation and training.

    Returns a trained model in form of a SparkML Model, which is also a Transformer.
    '''
    @keyword_only
    def __init__(self, keras_model_config=None, featuresCol=None, labelCol=None, optimizer_config=None, mode=None,
                 frequency=None, num_workers=None, nb_epoch=None, batch_size=None, verbose=None, validation_split=None,
                 categorical=None, nb_classes=None):
        super(ElephasEstimator, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, keras_model_config=None, featuresCol=None, labelCol=None, optimizer_config=None, mode=None,
                   frequency=None, num_workers=None, nb_epoch=None, batch_size=None, verbose=None,
                   validation_split=None, categorical=None, nb_classes=None):
        '''
        Set all provided parameters, otherwise set defaults
        '''
        kwargs = self.set_params._input_kwargs
        return self._set(**kwargs)

    def _fit(self, df):
        '''
        Private fit method of the Estimator, which trains the model.
        '''
        simple_rdd = df_to_simple_rdd(df, categorical=self.get_categorical_labels(), nb_classes=self.get_nb_classes(),
                                      featuresCol=self.getFeaturesCol(), labelCol=self.getLabelCol())
        simple_rdd = simple_rdd.repartition(self.get_num_workers())
        optimizer = None
        if self.get_optimizer_config() is not None:
            optimizer = get(self.get_optimizer_config()['name'], self.get_optimizer_config())

        spark_model = SparkModel(simple_rdd.ctx, model_from_yaml(self.get_keras_model_config()), optimizer=optimizer,
                                 mode=self.get_mode(), frequency=self.get_frequency(),
                                 num_workers=self.get_num_workers())
        spark_model.train(simple_rdd)

        model_weights = spark_model.master_network.get_weights()
        weights = simple_rdd.ctx.broadcast(model_weights)
        return ElephasTransformer(inputCol=self.getFeaturesCol(),
                                  keras_model_config=spark_model.master_network.to_yaml(),
                                  weights=weights)


class ElephasTransformer(Model, HasKerasModelConfig, HasFeaturesCol, HasInputCol, HasOutputCol):
    '''
    SparkML Transformer implementation. Contains a trained model,
    with which new feature data can be transformed into labels.
    '''
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, keras_model_config=None, weights=None):
        super(ElephasTransformer, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.weights = kwargs.pop('weights')
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None, outputCol=None, keras_model_config=None):
        '''
        Set all provided parameters, otherwise set defaults
        '''
        kwargs = self.set_params._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        '''
        Private transform method of a Transformer. This serves as batch-prediction method for our purposes.
        '''
        rdd = df.rdd
        features = np.asarray(rdd.map(lambda x: from_vector(x.features)).collect())
        # Note that we collect, since executing this on the rdd would require model serialization once again
        model = model_from_yaml(self.get_keras_model_config())
        model.set_weights(self.weights.value)
        predictions = rdd.ctx.parallelize(model.predict_classes(features))
        results_rdd = rdd.zip(predictions).map(lambda pair: Row(features=to_vector(pair[0].features),
                                               label=pair[0].label, prediction=float(pair[1])))
        results_df = df.sql_ctx.createDataFrame(results_rdd)
        return results_df
