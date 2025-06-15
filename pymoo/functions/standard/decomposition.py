"""
Standard Python implementations of decomposition functions.
"""

import numpy as np


def calc_distance_to_weights(F, weights, utopian_point=None):
    """Calculate distance to weights for decomposition methods."""
    norm = np.linalg.norm(weights, axis=1)

    if utopian_point is not None:
        F = F - utopian_point

    d1 = (F * weights).sum(axis=1) / norm
    d2 = np.linalg.norm(F - (d1[:, None] * weights / norm[:, None]), axis=1)

    return d1, d2