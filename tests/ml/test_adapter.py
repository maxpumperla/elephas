import numpy as np
from elephas.ml import adapter
import pytest
pytest.mark.usefixtures("spark_context")


def test_to_data_frame(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[2.0], [1.0]])

    data_frame = adapter.to_data_frame(
        spark_context, features, labels, categorical=False)
    assert data_frame.count() == 2


def test_to_data_frame_cat(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[0, 0, 1.0], [0, 1.0, 0]])

    data_frame = adapter.to_data_frame(
        spark_context, features, labels, categorical=True)
    assert data_frame.count() == 2


def test_from_data_frame(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[2.0], [1.0]]).reshape((2,))

    data_frame = adapter.to_data_frame(
        spark_context, features, labels, categorical=False)

    x, y = adapter.from_data_frame(data_frame, categorical=False)
    assert features.shape == x.shape
    assert labels.shape == y.shape


def test_from_data_frame_cat(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[0, 0, 1.0], [0, 1.0, 0]])

    data_frame = adapter.to_data_frame(
        spark_context, features, labels, categorical=True)

    x, y = adapter.from_data_frame(data_frame, categorical=True, nb_classes=3)
    assert features.shape == x.shape
    assert labels.shape == y.shape


def test_df_to_simple_rdd(spark_context):
    features = np.ones((2, 10))
    labels = np.asarray([[2.0], [1.0]]).reshape((2,))

    data_frame = adapter.to_data_frame(
        spark_context, features, labels, categorical=False)

    rdd = adapter.df_to_simple_rdd(data_frame, False)
    assert rdd.count() == 2
