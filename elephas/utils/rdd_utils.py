from __future__ import absolute_import

from pyspark.mllib.regression import LabeledPoint
import numpy as np

from ..mllib.adapter import to_vector, from_vector


def to_simple_rdd(sc, features, labels):
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)


def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):
        if categorical:
            lp = LabeledPoint(np.argmax(y), to_vector(x))
        else:
            lp = LabeledPoint(y, to_vector(x))
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)


def from_labeled_point(rdd, categorical=False, nb_classes=None):
    features = np.asarray(rdd.map(lambda lp: from_vector(lp.features)).collect())
    labels = np.asarray(rdd.map(lambda lp: lp.label).collect(), dtype='int32')
    if categorical:
        if not nb_classes:
            nb_classes = np.max(labels)+1
        temp = np.zeros((len(labels), nb_classes))
        for i in range(len(labels)):
            temp[i, labels[i]] = 1.
        labels = temp
    return features, labels


def encode_label(label, nb_classes):
    encoded = np.zeros(nb_classes)
    encoded[label] = 1.
    return encoded


def lp_to_simple_rdd(lp_rdd, categorical=False, nb_classes=None):
    if categorical:
        if not nb_classes:
            labels = np.asarray(lp_rdd.map(lambda lp: lp.label).collect(), dtype='int32')
            nb_classes = np.max(labels)+1
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features), encode_label(lp.label, nb_classes)))
    else:
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features), lp.label))
    return rdd
