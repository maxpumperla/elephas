import pytest
import numpy as np
from elephas.utils import functional_utils

pytest.mark.usefixtures("spark_context")


def test_add_params():
    p1 = [np.ones((5, 5)) for _ in range(10)]
    p2 = [np.ones((5, 5)) for _ in range(10)]

    res = functional_utils.add_params(p1, p2)
    assert len(res) == 10
    for i in range(5):
        for j in range(5):
            assert res[0][i, j] == 2


def test_subtract_params():
    p1 = [np.ones((5, 5)) for _ in range(10)]
    p2 = [np.ones((5, 5)) for _ in range(10)]

    res = functional_utils.subtract_params(p1, p2)

    assert len(res) == 10
    for i in range(5):
        for j in range(5):
            assert res[0][i, j] == 0


def test_get_neutral():
    x = [np.ones((3, 4))]
    res = functional_utils.get_neutral(x)
    assert res[0].shape == x[0].shape
    assert res[0][0, 0] == 0


def test_divide_by():
    x = [np.ones((3, 4))]
    res = functional_utils.divide_by(x, num_workers=10)
    assert res[0].shape == x[0].shape
    assert res[0][0, 0] == 0.1
