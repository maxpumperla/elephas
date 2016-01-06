from __future__ import absolute_import

import numpy as np


def add_params(p1, p2):
    '''
    Add two lists of parameters
    '''
    res = []
    for x, y in zip(p1, p2):
        res.append(x+y)
    return res


def subtract_params(p1, p2):
    '''
    Subtract two lists of parameters
    '''
    res = []
    for x, y in zip(p1, p2):
        res.append(x-y)
    return res


def get_neutral(array):
    '''
    Get list of zero-valued numpy arrays for
    specified list of numpy arrays
    '''
    res = []
    for x in array:
        res.append(np.zeros_like(x))
    return res


def divide_by(array_list, num_workers):
    '''
    Divide a list of parameters by an integer num_workers.
    '''
    for i, x in enumerate(array_list):
        array_list[i] /= num_workers
    return array_list
