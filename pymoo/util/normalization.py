import numpy as np
from numpy.linalg import LinAlgError


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

def normalize_by_asf_interceptions__(F, non_dom, return_bounds=False):
    pass


def normalize_by_asf_interceptions_(x, return_bounds=False):

    # find the x_min point
    n_obj = x.shape[1]
    x_min = np.min(x, axis=0)
    # transform the objective that 0 means the best
    F = x - x_min

    # calculate the asf matrix
    asf = np.eye(n_obj)
    asf = asf + (asf == 0) * 1e-16

    # result matrix with the selected points
    S = np.zeros((n_obj, n_obj))

    # find for each direction the best
    for i in range(len(asf)):
        val = np.max(F / asf[i, :], axis=1) + 0.0001 * np.sum(F / asf[i, :])
        S[i, :] = F[np.argmin(val), :]

    try:
        b = np.ones(n_obj)
        A = np.linalg.solve(S, b)
        A = 1 / A
        A = A.T
    except LinAlgError:
        A = np.max(S, axis=0)

    F = F / A

    if not return_bounds:
        return F
    else:
        x_max = A + x_min
        return F, x_min, x_max


def normalize_by_asf_interceptions(X, first_front, prev_S=None, prev_asf=None, return_bounds=False):
    """

    Normalization by asf intercepts. The following is a series of checks required during the normalization procedure.

    Regarding normalization:
    At each generation we store the extreme points that are used to form hyperplane.
    If extreme points found at current generation have a smaller asf value then we use this point only and update
    the stored point, in-case stored point has smaller asf then we use stored point.

    Regarding finding intercepts:
    If the gaussian elimination fails we take nadir point as intercept value
    (nadir point is formed by taking maximum value from first front members across each dimension).
    In case gaussian elimination succeeds but some intercept value is negative then we replace that particular
    value with the corresponding nadir point value.
    Last check that we make is, if intercept value thus found is smaller than 1e-6 we replace that particular
    value with the maximum value across all population members (including last front members) in that particular dimension.

    Parameters
    ----------
    X : numpy.ndarray
        Points in the design space
    first_front :
        first_front: length of the first front, represents the first f indexes of x
    prev_S :
        Stored extreme points
    prev_asf :
        Stored asf values
    return_bounds :
        Flag to return bounds, if true, F_max an F_min of population are returned

    Returns
    -------

    pop :
        Normalized population N, asf and S matrices to store, optionally F_min, F_max

    """

    # find the x_min point
    n_obj = X.shape[1]
    # Ideal point
    x_min = np.min(X, axis=0)
    F = X - x_min

    # nadir point is calculated as the max of each objective in the FIRST front only
    nadir = np.max(F[range(first_front)], axis=0)

    # calculate the asf matrix
    asf = np.eye(n_obj)
    asf = asf + (asf == 0) * 1e-16

    # result matrix with the selected points
    S = np.zeros((n_obj, n_obj))
    new_asf = []

    # find for each direction the best
    for i in range(len(asf)):
        val = np.max(F / asf[i, :], axis=1) + 0.0001 * np.sum(F / asf[i, :])
        new_asf.append(val[np.argmin(val)])
        S[i, :] = F[np.argmin(val), :]

    # Calculate and update extreme points
    if prev_S is not None and prev_asf is not None:
        prev_asf, prev_S = recalculate(x_min, prev_asf, prev_S)
        new_asf, S = update_extreme_points(new_asf, S, prev_asf, prev_S)
    try:
        b = np.ones(n_obj)
        A = np.linalg.solve(S, b)
        # A[A==0] = 0.000001
        A = 1 / A
        A = A.T
    except LinAlgError:

        # If gaussian elimination fails we select the nadir point as the intercept
        A = nadir

    # If an intercept value is negative, replace it with the associated nadir value
    A[A < 0] = nadir[A < 0]
    # If an intercept value is smaller than 1e-6 then we replace
    # it with the max value across population members in that direction
    A[A < 1e-6] = F[:, A < 1e-6].max(axis=0)

    F = F / A

    # We undo the ideal point translation because these extreme points need to be stored in the original space
    S = S + x_min
    if not return_bounds:
        return F, new_asf, S
    else:
        x_max = A + x_min
        return F, new_asf, S, x_min, x_max


def recalculate(ideal, asf, S):
    """
    This function simply calculates the asf of a set of points S
    Variables are passed by assignment therefore the asf matrix and point matrix
    are updated in place.
    :param ideal: Ideal point of the population
    :param asf: asf matrix from previous generation
    :param S: set of points, (extreme points from previous generation)
    :return: Updates asf matrix and S matrix with new values
    """
    n_obj = len(asf)
    A = np.eye(n_obj)
    A = A + (A == 0) * 1e-16
    # retranslate extreme points to the new ideal point
    S -= ideal

    # ASF calculation and extreme point
    for i in range(len(asf)):
        val = np.max(S / A[i, :], axis=1) + 0.0001 * np.sum(S / A[i, :])
        asf[i] = val[np.argmin(val)]
        S[i, :] = S[np.argmin(val), :]
    return asf, S


def update_extreme_points(new_asf, S, prev_asf, prev_S):
    """
    Updates the extreme points for the current generation using new and stored extreme points
    :param new_asf: newly computed asf values for current generation
    :param S: newly computed extreme points for current generation
    :param prev_asf: stored asf values
    :param prev_S: stored extreme points
    :return: best asf values, and best extreme points
    """
    extreme_points = []
    min_asf = []

    for i in range(len(S)):
        if new_asf[i] < prev_asf[i]:
            extreme_points.append(S[i])
            min_asf.append(new_asf[i])
        else:
            extreme_points.append(prev_S[i])
            min_asf.append(prev_asf[i])
    return min_asf, extreme_points


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
