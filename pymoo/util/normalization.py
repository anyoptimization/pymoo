import numpy as np


def denormalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def normalize(x, x_min=None, x_max=None, return_bounds=False):
    if x_min is None:
        x_min = np.min(x, axis=0)
    if x_max is None:
        x_max = np.max(x, axis=0)

    denom = x_max - x_min
    denom += 1e-30

    res = (x - x_min) / denom
    if not return_bounds:
        return res
    else:
        return res, x_min, x_max


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
