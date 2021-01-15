import numpy as np


def add_params(param_list_left: list, param_list_right: list):
    """Add two lists of parameters one by one

    :param param_list_left: list of numpy arrays
    :param param_list_right: list of numpy arrays
    :return: list of numpy arrays
    """
    return [x + y for x, y in zip(param_list_left, param_list_right)]


def subtract_params(param_list_left: list, param_list_right: list):
    """Subtract two lists of parameters

    :param param_list_left: list of numpy arrays
    :param param_list_right: list of numpy arrays
    :return: list of numpy arrays
    """
    return [x - y for x, y in zip(param_list_left, param_list_right)]


def get_neutral(array_list: list):
    """Get list of zero-valued numpy arrays for
    specified list of numpy arrays

    :param array_list: list of numpy arrays
    :return: list of zeros of same shape as input
    """
    return [np.zeros_like(x) for x in array_list]


def divide_by(array_list: list, num_workers: int):
    """Divide a list of parameters by an integer num_workers.

    :param array_list:
    :param num_workers:
    :return:
    """
    return [x / num_workers for x in array_list]
