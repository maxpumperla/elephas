import pytest
import numpy as np
from elephas.utils import rdd_utils

pytest.mark.usefixtures("spark_context")


def test_to_simple_rdd(spark_context):
    features = np.ones((5, 10))
    labels = np.ones((5,))
    rdd = rdd_utils.to_simple_rdd(spark_context, features, labels)

    assert rdd.count() == 5
    first = rdd.first()
    assert first[0].shape == (10,)
    assert first[1] == 1.0


def test_to_labeled_rdd_categorical(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[0, 0, 1.0], [0, 1.0, 0]])
    lp_rdd = rdd_utils.to_labeled_point(spark_context, features, labels, True)
    assert lp_rdd.count() == 2
    first = lp_rdd.first()
    assert first.features.shape == (10,)
    assert first.label == 2.0


def test_to_labeled_rdd_not_categorical(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[2.0], [1.0]])
    lp_rdd = rdd_utils.to_labeled_point(spark_context, features, labels, False)
    assert lp_rdd.count() == 2
    first = lp_rdd.first()
    assert first.features.shape == (10,)
    assert first.label == 2.0


def test_from_labeled_rdd(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[2.0], [1.0]]).reshape((2,))
    lp_rdd = rdd_utils.to_labeled_point(spark_context, features, labels, False)

    x, y = rdd_utils.from_labeled_point(lp_rdd, False, None)
    assert x.shape == features.shape
    assert y.shape == labels.shape


def test_from_labeled_rdd_categorical(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[0, 0, 1.0], [0, 1.0, 0]])
    lp_rdd = rdd_utils.to_labeled_point(spark_context, features, labels, True)

    x, y = rdd_utils.from_labeled_point(lp_rdd, True, 3)
    assert x.shape == features.shape
    assert y.shape == labels.shape


def test_encode_label():
    label = 3
    nb_classes = 10
    encoded = rdd_utils.encode_label(label, nb_classes)
    assert len(encoded) == nb_classes
    for i in range(10):
        if i == label:
            assert encoded[i] == 1
        else:
            assert encoded[i] == 0


def test_lp_to_simple_rdd_categorical(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[0, 0, 1.0], [0, 1.0, 0]])
    lp_rdd = rdd_utils.to_labeled_point(spark_context, features, labels, True)

    rdd = rdd_utils.lp_to_simple_rdd(lp_rdd, categorical=True, nb_classes=3)
    assert rdd.first()[0].shape == (10,)
    assert rdd.first()[1].shape == (3,)


def test_lp_to_simple_rdd_not_categorical(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[2.0], [1.0]]).reshape((2,))
    lp_rdd = rdd_utils.to_labeled_point(spark_context, features, labels, False)

    rdd = rdd_utils.lp_to_simple_rdd(lp_rdd, categorical=False, nb_classes=3)
    assert rdd.first()[0].shape == (10,)
    assert rdd.first()[1] == 2.0
