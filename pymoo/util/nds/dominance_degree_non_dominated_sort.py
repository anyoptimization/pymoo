"""Module which implements Dominance Degree Approaches for Non-dominated Sorting.

For the original work see:
    DDA-NS https://ieeexplore.ieee.org/document/7469397
    DDA-ENS https://ieeexplore.ieee.org/document/9282978

Adapted from https://github.com/rsenwar/Non-Dominated-Sorting-Algorithms/tree/master
"""


import numpy as np


def construct_comp_matrix(vec: np.ndarray) -> np.ndarray:
    """
    const_comp_mat construct the comparison matrix from a row-vector w.
    """
    v = vec.squeeze()
    n = v.shape[-1]
    # a is the sorted vector in ascending order, and b is the index vector satisfying a = w(b)
    b = np.argsort(v, axis=-1, kind="quicksort")
    c = np.zeros(shape=(n, n), dtype=np.int32)

    # the elements of the b(0)-th row in C are all set to 1
    c[b[0], :] = 1

    for i in range(1, n):
        if v[b[i]] == v[b[i - 1]]:
            # the rows in C corresponding to the same elements in w are identical
            c[b[i]] = c[b[i - 1]]
        else:
            c[b[i], b[i:]] = 1

    return c


def construct_domination_matrix(f_scores: np.ndarray, **kwargs) -> np.ndarray:
    """
    construct_domination_matrix calculates the dominance degree matrix for a set of vectors.

    The dominance degree indicate the degree of dominance of a solution, which is the number of
    objectives for which it is the dominating solution.

    Parameters
    ----------
    f_scores : np.ndarray
        an N x M matrix of N (population size) objective function values for M objectives
    """
    d = np.zeros((f_scores.shape[0], f_scores.shape[0]), dtype=np.int32)
    for vec in np.split(f_scores, f_scores.shape[-1], axis=-1):
        d += construct_comp_matrix(vec)
    return d


def dda_ns(f_scores: np.ndarray, **kwargs) -> list[list[int]]:
    """
    dda_ns runs the DDA-NS algorithm.

    Parameters
    ----------
    f_scores : np.ndarray
        an N x M matrix of N (population size) objective function values for M objectives

    Returns
    -------
    list[list[int]]
        A list of members of each Pareto front. The index in the outer most list corresponds to the level in the Pareto front
        while the value in the inner-most list is the id of the member of the population belonging to that front.
    """
    d_mx = construct_domination_matrix(f_scores)

    # remove dominating solutions (D[i, j] == m && D[j, i] == M)
    d_mx = np.where(
        np.logical_and(
            np.equal(d_mx, f_scores.shape[-1]), np.equal(d_mx.T, f_scores.shape[-1])
        ),
        0,
        d_mx,
    )

    fronts = []
    count = 0
    while count < f_scores.shape[0]:
        front = []
        # Max(D) is the row vector containing the maximum elements from each column of D
        max_d = np.max(d_mx, axis=0)
        for i in range(f_scores.shape[0]):
            if 0 <= max_d[i] < f_scores.shape[-1]:
                # solution Yi belongs to current front
                front.append(i)
                count += 1
        for i in front:
            d_mx[i, :] = -1
            d_mx[:, i] = -1
        fronts.append(front)

    return fronts


def dda_ens(f_scores: np.ndarray, **kwargs) -> list[list[int]]:
    d_mx = construct_domination_matrix(f_scores)

    d_mx = np.where(
        np.logical_and(d_mx == f_scores.shape[-1], d_mx.T == f_scores.shape[-1]),
        0,
        d_mx,
    )

    fronts: list[list[int]] = []
    front_count = 0
    for s in np.lexsort(f_scores.T):
        isinserted = False
        for k in range(front_count):
            isdominated = any(d_mx[sd, s] == f_scores.shape[1] for sd in fronts[k])
            if not isdominated:
                fronts[k].append(s)
                isinserted = True
                break
        if not isinserted:
            front_count += 1
            fronts.append([s])
    return fronts


def dominance_degree_non_dominated_sort(f_scores: np.ndarray, strategy="efficient"):
    if strategy == "efficient":
        return dda_ens(f_scores)
    if strategy == "fast":
        return dda_ns(f_scores)
    raise ValueError("Invalid search strategy")
