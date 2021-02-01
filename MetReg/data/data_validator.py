import numpy as np


class Data_validator():

    def __init__(self):
        pass


def _assert_all_finite(X, allow_nan=False,):
    """Throw a ValueError if X contains NaN or infinity.

    Args:
        X ([type]): [description]
        allow_nan (bool, optional): [description]. Defaults to False.
    """
    pass


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    """
    pass


def check_array(array):
    """input validataion on an array.

    Args:
        array ([type]): [description]
    """
    pass


def check_X_y(X, y):
    """Input validation for all standard regression.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Args:
        X ([type]): [description]
        y ([type]): [description]
    """
    pass

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Args:
        seed ([type]): [description]
    """
    pass

def check_non_negative(X,):
    """Check if there is any negative value in an array.

    Args:
        X ([type]): [description]
    """
    pass

def check_target_sparse(y):
    """target data as sparse is not supported

    Args:
        y ([type]): [description]
    """
    pass

def _check_numpy(obj):
    """Check obj is numpy matrix"""
    if type(obj) != np.ndarray:
        obj = np.array(obj)
    return obj
