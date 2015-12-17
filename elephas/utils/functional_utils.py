from __future__ import absolute_import

import numpy as np
from six.moves import range


def add_params(p1, p2):
    res = []
    for x, y in zip(p1, p2):
        res.append(x+y)
    return res


def subtract_params(p1, p2):
    res = []
    for x, y in zip(p1, p2):
        res.append(x-y)
    return res


def get_neutral(array):
    res = []
    for x in array:
        res.append(np.zeros_like(x))
    return res


def divide_by(array_list, num_workers):
    for i in range(len(array_list)):
        array_list[i] /= num_workers
    return array_list
