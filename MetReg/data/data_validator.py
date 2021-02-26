import numpy as np
import math


class Data_validator():
    """A class to pre-validator dataset, in this regard, the format of the data
    is checked, also the dimensionality and rationality. and if applicable, 
    features are encoded (one-hot encoding).
    """

    def __init__(self,
                 X_y_dims: list = [2, 1]):
        if isinstance(X_y_dims, (list, tuple, np.array)):
            self.X_y_dims = X_y_dims
        else:
            raise ValueError("X_y_dims is list, tuple or numpy array")

    def fit(self, X, y):
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
        if y is None:
            raise ValueError("y can't be None for `check_X_y`")

        if _check_numpy(X):
            pass

    def _assert_all_finite(self, X, allow_nan=False,):
        """Throw a ValueError if X contains NaN or infinity.

        Args:
            X ([type]): [description]
            allow_nan (bool, optional): [description]. Defaults to False.
        """
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError('X contain NaN or Inf')

    def check_consistent_length(self, *arrays):
        """Check that all arrays have consistent first dimensions."""
        lengths = [X.shape[0] for X in arrays if X is not None]
        uniques = np.unique(lengths)
        if len(uniques) > 1:
            raise ValueError(
                "Found input variables with inconsistent"
                " numbers of samples: {}".format([int(l) for l in lengths]))

    def _check_numpy(self, obj):
        """Check obj is numpy matrix"""
        if type(obj) != np.ndarray:
            obj = np.array(obj)
        return obj

    def check_non_negative(self, X,):
        """Check if there is any negative value in an array.

        Args:
            X ([type]): [description]
        """
        pass

    def check_target_sparse(self, y):
        """target data as sparse is not supported

        Args:
            y ([type]): [description]
        """
        pass


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Args:
        seed ([type]): [description]
    """
    pass
