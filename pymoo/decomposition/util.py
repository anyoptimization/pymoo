"""Decomposition utility functions."""

import numpy as np


def calc_distance_to_weights(F, weights, utopian_point=None):
    """Calculate distance components to weight vectors.

    Args:
        F: Objective values array.
        weights: Weight vectors array.
        utopian_point: Optional utopian point for normalization.

    Returns:
        Tuple of (d1, d2) distance components.
    """
    norm = np.linalg.norm(weights, axis=1)

    if utopian_point is not None:
        F = F - utopian_point

    d1 = (F * weights).sum(axis=1) / norm
    d2 = np.linalg.norm(F - (d1[:, None] * weights / norm[:, None]), axis=1)

    return d1, d2
