from __future__ import absolute_import

import numpy as np
from six.moves import zip


def add_params(param_list_left, param_list_right):
    """Add two lists of parameters one by one

    :param param_list_left: list of numpy arrays
    :param param_list_right: list of numpy arrays
    :return: list of numpy arrays
    """
    res = []
    for x, y in zip(param_list_left, param_list_right):
        res.append(x + y)
    return res


def subtract_params(param_list_left, param_list_right):
    """Subtract two lists of parameters

    :param param_list_left: list of numpy arrays
    :param param_list_right: list of numpy arrays
    :return: list of numpy arrays
    """
    res = []
    for x, y in zip(param_list_left, param_list_right):
        res.append(x - y)
    return res


def get_neutral(array_list):
    """Get list of zero-valued numpy arrays for
    specified list of numpy arrays

    :param array_list: list of numpy arrays
    :return: list of zeros of same shape as input
    """
    res = []
    for x in array_list:
        res.append(np.zeros_like(x))
    return res


def divide_by(array_list, num_workers):
    """Divide a list of parameters by an integer num_workers.

    :param array_list:
    :param num_workers:
    :return:
    """
    for i, x in enumerate(array_list):
        array_list[i] /= num_workers
    return array_list
