from __future__ import absolute_import

from pyspark.mllib.linalg import Matrices, Vectors


def from_matrix(matrix):
    return matrix.toArray()


def to_matrix(np_array):
    if len(np_array.shape) == 2:
        return Matrices.dense(np_array.shape[0],
                              np_array.shape[1],
                              np_array.ravel())
    else:
        raise Exception("""An MLLib Matrix can only be created
                        from a two-dimensional numpy array""")


def from_vector(vector):
    return vector.array


def to_vector(np_array):
    if len(np_array.shape) == 1:
        return Vectors.dense(np_array)
    else:
        raise Exception("""An MLLib Vector can only be created
                        from a one-dimensional numpy array""")
