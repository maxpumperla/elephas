import numpy as np
from elephas.mllib.adapter import *
from pyspark.mllib.linalg import Matrices, Vectors


def test_to_matrix():
    x = np.ones((4, 2))
    mat = to_matrix(x)
    assert mat.numRows == 4
    assert mat.numCols == 2


def test_from_matrix():
    mat = Matrices.dense(1, 2, [13, 37])
    x = from_matrix(mat)
    assert x.shape == (1, 2)


def test_from_vector():
    x = np.ones((3,))
    vector = to_vector(x)
    assert len(vector) == 3


def test_to_vector():
    vector = Vectors.dense([4, 2])
    x = from_vector(vector)
    assert x.shape == (2,)
