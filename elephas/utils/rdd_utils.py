from pyspark import RDD, SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np

from ..mllib.adapter import to_vector, from_vector
try:
    from elephas.java import java_classes
    from elephas.java.ndarray import ndarray
except Exception:
    print("WARNING")


def to_java_rdd(jsc, features, labels, batch_size):
    """Convert numpy features and labels into a JavaRDD of
    DL4J DataSet type.

    :param jsc: JavaSparkContext from pyjnius
    :param features: numpy array with features
    :param labels: numpy array with labels:
    :return: JavaRDD<DataSet>
    """
    data_sets = java_classes.ArrayList()
    num_batches = int(len(features) / batch_size)
    for i in range(num_batches):
        xi = ndarray(features[:batch_size].copy())
        yi = ndarray(labels[:batch_size].copy())
        data_set = java_classes.DataSet(xi.array, yi.array)
        data_sets.add(data_set)
        features = features[batch_size:]
        labels = labels[batch_size:]

    return jsc.parallelize(data_sets)


def to_simple_rdd(sc: SparkContext, features: np.array, labels: np.array):
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)


def to_labeled_point(sc: SparkContext, features: np.array, labels: np.array, categorical: bool = False):
    """Convert numpy arrays of features and labels into
    a LabeledPoint RDD for MLlib and ML integration.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :param categorical: boolean, whether labels are already one-hot encoded or not
    :return: LabeledPoint RDD with features and labels
    """
    labeled_points = []
    for x, y in zip(features, labels):
        if categorical:
            lp = LabeledPoint(np.argmax(y), to_vector(x))
        else:
            lp = LabeledPoint(y, to_vector(x))
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)


def from_labeled_point(rdd: RDD, categorical: bool = False, nb_classes: int = None):
    """Convert a LabeledPoint RDD back to a pair of numpy arrays

    :param rdd: LabeledPoint RDD
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: optional int, indicating the number of class labels
    :return: pair of numpy arrays, features and labels
    """
    features = np.asarray(
        rdd.map(lambda lp: from_vector(lp.features)).collect())
    labels = np.asarray(rdd.map(lambda lp: lp.label).collect(), dtype='int32')
    if categorical:
        if not nb_classes:
            nb_classes = np.max(labels) + 1
        temp = np.zeros((len(labels), nb_classes))
        for i, label in enumerate(labels):
            temp[i, label] = 1.
        labels = temp
    return features, labels


def encode_label(label: np.array, nb_classes: int):
    """One-hot encoding of a single label

    :param label: class label (int or double without floating point digits)
    :param nb_classes: int, number of total classes
    :return: one-hot encoded vector
    """
    encoded = np.zeros(nb_classes)
    encoded[int(label)] = 1.
    return encoded


def lp_to_simple_rdd(lp_rdd: RDD, categorical: bool = False, nb_classes: int = None):
    """Convert a LabeledPoint RDD into an RDD of feature-label pairs

    :param lp_rdd: LabeledPoint RDD of features and labels
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: int, number of total classes
    :return: Spark RDD with feature-label pairs
    """
    if categorical:
        if not nb_classes:
            labels = np.asarray(lp_rdd.map(
                lambda lp: lp.label).collect(), dtype='int32')
            nb_classes = np.max(labels) + 1
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features),
                                     encode_label(lp.label, nb_classes)))
    else:
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features), lp.label))
    return rdd
