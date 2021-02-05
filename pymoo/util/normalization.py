from abc import abstractmethod

import numpy as np

from pymoo.util.misc import logical_op, replace_nan_by


# ---------------------------------------------------------------------------------------------------------
# Object Oriented Interface
# ---------------------------------------------------------------------------------------------------------

# ---- Abstract Class
class Normalization:

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, X):
        pass


# ---- Useful if normalization is optional - can be simply disabled by using this object
class NoNormalization(Normalization):

    def forward(self, X):
        return X

    def backward(self, X):
        return X


# ---- Normalizes between zero and one given bounds or estimating them
class ZeroToOneNormalization(Normalization):

    def __init__(self, xl, xu) -> None:
        super().__init__()

        # determines whether the normalization is enabled at all
        self.disabled = True if (xl is None or xu is None) else False
        if self.disabled:
            return

        # whenever xl or xu is nan do no normalization at all there - this is the mask for that
        is_nan = np.logical_or(np.isnan(xl), np.isnan(xu))

        # if neither is nan than xu must be greater or equal than xl
        assert np.all(np.logical_or(xu >= xl, is_nan)), "xl must be less or equal than xu."

        # only work on copies because nan values are replaced now
        xl, xu = xl.copy(), xu.copy()
        xl[is_nan] = 0.0
        xu[is_nan] = 1.0

        # calculate the range which will be divided by later on
        _range = (xu - xl).astype(np.float)
        range_is_zero = (xl == xu)
        _range[range_is_zero] = np.nan

        self.xl = xl
        self.xu = xu
        self._range = _range
        self.ignore = is_nan
        self.range_is_zero = range_is_zero

    def forward(self, X):
        if self.disabled:
            return X

        xl, xu, _range, ignore, range_is_zero = self.xl, self.xu, self._range, self.ignore, self.range_is_zero

        # do the normalization
        N = (X - xl) / _range

        # set the values which should not have been normalized to the input values X
        N[..., ignore] = X[..., ignore]

        # if the range is zero simply use a zero here
        N[..., range_is_zero] = 0.0

        return N

    def backward(self, N):
        if self.disabled:
            return N

        xl, _range, ignore, range_is_zero = self.xl, self._range, self.ignore, self.range_is_zero

        X = N * _range + xl
        X[..., ignore] = N[..., ignore]
        X[..., range_is_zero] = xl[range_is_zero]

        return X


# ---------------------------------------------------------------------------------------------------------
# Functional Interface
# ---------------------------------------------------------------------------------------------------------


def normalize(x, xl=None, xu=None, return_bounds=False, estimate_bounds_if_none=True):

    if estimate_bounds_if_none:
        if xl is None:
            xl = np.min(x, axis=0)
        if xu is None:
            xu = np.max(x, axis=0)

    norm = ZeroToOneNormalization(xl, xu)
    x = norm.forward(x)

    if not return_bounds:
        return x
    else:
        return x, norm.xl, norm.xu


def denormalize(x, xl, xu):
    return ZeroToOneNormalization(xl, xu).backward(x)


def standardize(x, return_bounds=False):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # standardize
    val = (x - mean) / std

    if not return_bounds:
        return val
    else:
        return val, mean, std


def destandardize(x, mean, std):
    return (x * std) + mean
