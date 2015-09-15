from __future__ import absolute_import

from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint
from ..utils.rdd_utils import from_labeled_point, to_labeled_point, lp_to_simple_rdd

import numpy as np

def to_data_frame(sc, features, labels, categorical=False):
    lp_rdd = to_labeled_point(sc, features, labels, categorical)
    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(lp_rdd)
    return df

def from_data_frame(df, categorical=False, nb_classes=None):
    lp_rdd = df.rdd.map(lambda row: LabeledPoint(row.label, row.features))
    result = from_labeled_point(lp_rdd, categorical, nb_classes)
    return result

def df_to_simple_rdd(df, categorical=False, nb_classes=None):
    lp_rdd = df.rdd.map(lambda row: LabeledPoint(row.label, row.features))
    rdd = lp_to_simple_rdd(lp_rdd, categorical, nb_classes)
    return rdd
