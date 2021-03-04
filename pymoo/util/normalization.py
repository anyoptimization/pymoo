from abc import abstractmethod

import numpy as np


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

    def __init__(self, xl=None, xu=None) -> None:
        super().__init__()

        # if both are None we are basically done because normalization is disabled
        if xl is None and xu is None:
            self.xl, self.xu = None, None
            return

        # if not set simply fall back no nan values
        if xl is None:
            xl = np.full_like(xu, np.nan)
        if xu is None:
            xu = np.full_like(xl, np.nan)

        xl, xu = np.copy(xl).astype(float), np.copy(xu).astype(float)

        # if both are equal then set the upper bound to none (always the 0 or lower bound will be returned then)
        xu[xl == xu] = np.nan

        # store the lower and upper bounds
        self.xl, self.xu = xl, xu

        # check out when the input values are nan
        xl_nan, xu_nan = np.isnan(xl), np.isnan(xu)

        # now create all the masks that are necessary
        self.xl_only, self.xu_only = np.logical_and(~xl_nan, xu_nan), np.logical_and(xl_nan, ~xu_nan)
        self.both_nan = np.logical_and(np.isnan(xl), np.isnan(xu))
        self.neither_nan = ~self.both_nan

        # if neither is nan than xu must be greater or equal than xl
        any_nan = np.logical_or(np.isnan(xl), np.isnan(xu))
        assert np.all(np.logical_or(xu >= xl, any_nan)), "xl must be less or equal than xu."

    def forward(self, X):
        if X is None or (self.xl is None and self.xu is None):
            return X

        xl, xu, xl_only, xu_only = self.xl, self.xu, self.xl_only, self.xu_only
        both_nan, neither_nan = self.both_nan, self.neither_nan

        # simple copy the input
        N = np.copy(X)

        # normalize between zero and one if neither of them is nan
        N[..., neither_nan] = (X[..., neither_nan] - xl[neither_nan]) / (xu[neither_nan] - xl[neither_nan])

        N[..., xl_only] = X[..., xl_only] - xl[xl_only]

        N[..., xu_only] = 1.0 - (xu[xu_only] - X[..., xu_only])

        return N

    def backward(self, N):
        if N is None or (self.xl is None and self.xu is None):
            return N

        xl, xu, xl_only, xu_only = self.xl, self.xu, self.xl_only, self.xu_only
        both_nan, neither_nan = self.both_nan, self.neither_nan

        X = N.copy()
        X[..., neither_nan] = xl[neither_nan] + N[..., neither_nan] * (xu[neither_nan] - xl[neither_nan])

        X[..., xl_only] = N[..., xl_only] + xl[xl_only]

        X[..., xu_only] = xu[xu_only] - (1.0 - N[..., xu_only])

        return X


# ---------------------------------------------------------------------------------------------------------
# Functional Interface
# ---------------------------------------------------------------------------------------------------------


def normalize(X, xl=None, xu=None, return_bounds=False, estimate_bounds_if_none=True):
    if estimate_bounds_if_none:
        if xl is None:
            xl = np.min(X, axis=0)
        if xu is None:
            xu = np.max(X, axis=0)

    if isinstance(xl, float) or isinstance(xl, int):
        xl = np.full(X.shape[-1], xl)

    if isinstance(xu, float) or isinstance(xu, int):
        xu = np.full(X.shape[-1], xu)

    norm = ZeroToOneNormalization(xl, xu)
    X = norm.forward(X)

    if not return_bounds:
        return X
    else:
        return X, norm.xl, norm.xu


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


# ---------------------------------------------------------------------------------------------------------
# Pre Normalization
# A class inheriting from it can use the in-built feature of normalizing
# ---------------------------------------------------------------------------------------------------------


class PreNormalization:

    def __init__(self, zero_to_one=False, ideal=None, nadir=None, **kwargs):

        # normalization related stuff if that should be performed beforehand
        self.ideal, self.nadir = ideal, nadir

        if zero_to_one:
            assert self.ideal is not None and self.nadir is not None, "For normalization either provide pf or bounds!"

            n_dim = len(self.ideal)
            self.normalization = ZeroToOneNormalization(self.ideal, self.nadir)

            # now the ideal and nadir points have change to only zeros and ones
            self.ideal, self.nadir = np.zeros(n_dim), np.ones(n_dim)

        else:
            self.normalization = NoNormalization()

    def do(self, *args, **kwargs):
        pass
