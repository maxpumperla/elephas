from __future__ import absolute_import, print_function

import numpy as np

from pyspark.sql import Row
from pyspark.ml import Estimator, Transformer

from .spark_model import SparkModel
from .utils.rdd_utils import from_vector, to_vector
from .ml.adapter import df_to_simple_rdd


class ElephasEstimator(SparkModel, Estimator):
    '''
    SparkML Estimator implementation of an elephas model.
    This estimator takes all relevant arguments for model
    compilation and training.

    Returns a trained model in form of a SparkML Transformer.
    '''
    def __init__(self,
                 sc,
                 master_network,
                 optimizer=None,
                 mode='asynchronous',
                 frequency='epoch',
                 num_workers=4,
                 nb_epoch=20,
                 batch_size=32,
                 verbose=0,
                 validation_split=0.1,
                 categorical=True,
                 nb_classes=None):
        super(ElephasEstimator, self).__init__(sc,
                                               master_network,
                                               optimizer,
                                               mode,
                                               frequency,
                                               num_workers)
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.categorical = categorical
        self.nb_classes = nb_classes

    def _fit(self, df):
        ''' Private fit method of the Estimator '''
        simple_rdd = df_to_simple_rdd(df, categorical=self.categorical,
                                      nb_classes=self.nb_classes)
        simple_rdd = simple_rdd.repartition(self.num_workers)
        self.train(simple_rdd, nb_epoch=self.nb_epoch,
                   batch_size=self.batch_size, verbose=self.verbose,
                   validation_split=self.validation_split)
        return ElephasTransformer(self.spark_context, self.master_network)


class ElephasTransformer(SparkModel, Transformer):
    '''
    SparkML Transformer implementation. Contains a trained model,
    with which new feature data can be transformed into labels.
    '''

    def __init__(self, sc, master_network):
        super(ElephasTransformer, self).__init__(sc, master_network)

    def _transform(self, df):
        ''' Private transform method of a Transformer '''
        rdd = df.rdd
        features = np.asarray(rdd.map(lambda x: from_vector(x.features)).collect())
        # Note that we collect, since executing this on the
        # rdd would require model serialization once again
        predictions = self.spark_context.parallelize(self.master_network.predict_classes(features))
        results_rdd = rdd.zip(predictions).map(lambda pair: Row(features=to_vector(pair[0].features),
                                               label=pair[0].label, prediction=float(pair[1])))
        results_df = df.sql_ctx.createDataFrame(results_rdd)
        return results_df
