import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("spark_context")


def test_that_requires_sc(spark_context):
    assert spark_context.parallelize(np.zeros((10, 10))).count() == 10
