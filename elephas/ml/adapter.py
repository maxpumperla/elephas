from __future__ import absolute_import

from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint
from ..utils.rdd_utils import from_labeled_point, to_labeled_point, lp_to_simple_rdd


def to_data_frame(sc, features, labels, categorical=False):
    '''
    Convert numpy arrays of features and labels into Spark DataFrame
    '''
    lp_rdd = to_labeled_point(sc, features, labels, categorical)
    sql_context = SQLContext(sc)
    df = sql_context.createDataFrame(lp_rdd)
    return df


def from_data_frame(df, categorical=False, nb_classes=None):
    '''
    Convert DataFrame back to pair of numpy arrays
    '''
    lp_rdd = df.rdd.map(lambda row: LabeledPoint(row.label, row.features))
    features, labels = from_labeled_point(lp_rdd, categorical, nb_classes)
    return features, labels


def df_to_simple_rdd(df, categorical=False, nb_classes=None, featuresCol='features', labelCol='label'):
    '''
    Convert DataFrame into RDD of pairs
    '''
    sqlContext = df.sql_ctx
    sqlContext.registerDataFrameAsTable(df, "temp_table")
    selected_df = sqlContext.sql("SELECT {0} AS features, {1} as label from temp_table".format(featuresCol, labelCol))
    lp_rdd = selected_df.rdd.map(lambda row: LabeledPoint(row.label, row.features))
    rdd = lp_to_simple_rdd(lp_rdd, categorical, nb_classes)
    return rdd
