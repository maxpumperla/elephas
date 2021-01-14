from pyspark.mllib.linalg import Matrices, Vectors


def from_matrix(matrix):
    """Convert MLlib Matrix to numpy array """
    return matrix.toArray()


def to_matrix(np_array):
    """Convert numpy array to MLlib Matrix
    """
    if len(np_array.shape) == 2:
        return Matrices.dense(np_array.shape[0],
                              np_array.shape[1],
                              np_array.ravel())
    else:
        raise Exception("An MLLib Matrix can only be created from a two-dimensional " +
                        "numpy array, got {}".format(len(np_array.shape)))


def from_vector(vector):
    """Convert MLlib Vector to numpy array
    """
    return vector.toArray()


def to_vector(np_array):
    """Convert numpy array to MLlib Vector
    """
    if len(np_array.shape) == 1:
        return Vectors.dense(np_array)
    else:
        raise Exception("An MLLib Vector can only be created from a one-dimensional " +
                        "numpy array, got {}".format(len(np_array.shape)))
