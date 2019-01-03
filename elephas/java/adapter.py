import ctypes
import numpy as np
try:
    from elephas.java import java_classes
except:
    pass


def get_context_dtype():
    """Returns the nd4j dtype
    """
    dtype = java_classes.DataTypeUtil.getDtypeFromContext()
    return java_classes.DataTypeUtil.getDTypeForName(dtype)


def to_numpy(nd4j_array):
    """ Convert an ND4J array to a numpy array
    :param nd4j_array:
    :return:
    """
    buff = nd4j_array.data()
    address = buff.pointer().address()
    type_name = java_classes.DataTypeUtil.getDtypeFromContext()
    data_type = java_classes.DataTypeUtil.getDTypeForName(type_name)
    mapping = {
        'double': ctypes.c_double,
        'float': ctypes.c_float
    }
    Pointer = ctypes.POINTER(mapping[data_type])
    pointer = ctypes.cast(address, Pointer)
    np_array = np.ctypeslib.as_array(pointer, tuple(nd4j_array.shape()))
    return np_array


def retrieve_keras_weights(java_model):
    """For a previously imported Keras model, after training it with DL4J Spark,
    we want to set the resulting weights back to the original Keras model.

    :param java_model: DL4J model (MultiLayerNetwork or ComputationGraph
    :return: list of numpy arrays in correct order for model.set_weights(...) of a corresponding Keras model
    """
    weights = []
    layers = java_model.getLayers()
    for layer in layers:
        params = layer.paramTable()
        keys = params.keySet()
        key_list = java_classes.ArrayList(keys)
        for key in key_list:
            weight = params.get(key)
            np_weight = np.squeeze(to_numpy(weight))
            weights.append(np_weight)
    return weights
