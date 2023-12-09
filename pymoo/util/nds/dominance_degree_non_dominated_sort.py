"""Module which implements Dominance Degree Approaches for Non-dominated Sorting.

For the original work see:
    DDA-NS https://ieeexplore.ieee.org/document/7469397
    DDA-ENS https://ieeexplore.ieee.org/document/9282978

Adapted from https://github.com/rsenwar/Non-Dominated-Sorting-Algorithms/tree/master
"""


from typing import Literal, List
import numpy as np


def construct_comp_matrix(vec: np.ndarray, sorted_idx: np.ndarray) -> np.ndarray:
    """
    const_comp_mat construct the comparison matrix from a row-vector vec.

    Parameters
    ----------
    vec : np.ndarray
        The vector of scores for the population on a single objective
    sorted_idx : np.ndarray
        The indices which would sort `vec`

    Returns
    -------
    np.ndarray
        The comparison matrix indicating whether each member in the population dominates the other member for the
        objective in `vec`
    """
    n = vec.shape[0]
    c = np.zeros(shape=(n, n), dtype=np.int32)

    # the elements of the b(0)-th row in C are all set to 1
    c[sorted_idx[0], :] = 1

    for i in range(1, n):
        if vec[sorted_idx[i]] == vec[sorted_idx[i - 1]]:
            # the rows in C corresponding to the same elements in w are identical
            c[sorted_idx[i]] = c[sorted_idx[i - 1]]
        else:
            c[sorted_idx[i], sorted_idx[i:]] = 1

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
    b = np.apply_over_axes(np.argsort, f_scores, axes=0)
    for vec, srt in zip(f_scores.T, b.T):
        d += construct_comp_matrix(vec, srt)
    d = np.where(
        np.logical_and(d == f_scores.shape[-1], d.T == f_scores.shape[-1]), 0, d
    )
    return d


def dda_ns(f_scores: np.ndarray, **kwargs) -> List[List[int]]:
    """
    dda_ns runs the DDA-NS algorithm.

    Parameters
    ----------
    f_scores : np.ndarray
        an N x M matrix of N (population size) objective function values for M objectives

    Returns
    -------
    List[List[int]]
        A list of members of each Pareto front. The index in the outer most list corresponds to the level in the Pareto front
        while the value in the inner-most list is the id of the member of the population belonging to that front.
    """
    d_mx = construct_domination_matrix(f_scores)
    max_d = np.empty((f_scores.shape[0],), dtype=np.int32)

    fronts = []
    count = 0
    while count < f_scores.shape[0]:
        # Max(D) is the row vector containing the maximum elements from each column of D
        np.max(d_mx, out=max_d, axis=0)
        front = [i for i, m_d in enumerate(max_d) if 0 <= m_d < f_scores.shape[-1]]
        count += len(front)
        d_mx[front] = -1
        d_mx[:, front] = -1
        fronts.append(front)

    return fronts


def dda_ens(f_scores: np.ndarray, **kwargs) -> List[List[int]]:
    """
    dda_ens runs the DDA-ENS (efficient DDA) algorithm

    Parameters
    ----------
    f_scores : np.ndarray
        The N x M matrix of N (population size) objective function values for M objectives

    Returns
    -------
    List[List[int]]
        an N x M matrix of N (population size) objective function values for M objectives
    """
    d_mx = construct_domination_matrix(f_scores)

    fronts: List[List[int]] = []
    for s in np.lexsort(f_scores.T):
        isinserted = False
        for fk in fronts:
            if not (d_mx[fk, s] == f_scores.shape[1]).any():
                fk.append(s)
                isinserted = True
                break
        if not isinserted:
            fronts.append([s])
    return fronts


def dominance_degree_non_dominated_sort(
    f_scores: np.ndarray, strategy: Literal["efficient", "fast"] = "efficient"
) -> List[List[int]]:
    """
    dominance_degree_non_dominated_sort performs the non-dominating sort with the specified algorithm

    Parameters
    ----------
    f_scores : np.ndarray
        The N x M matrix of N (population size) objective function values for M objectives
    strategy : Literal["efficient", "fast"], optional
        The dominance degree algorithm to use, by default "efficient"

    Returns
    -------
    List[List[int]]
        A list of members of each Pareto front. The index in the outer most list corresponds to the level in the Pareto front
        while the value in the inner-most list is the id of the member of the population belonging to that front.

    Raises
    ------
    ValueError
        If an invalid strategy is specified
    """
    if strategy == "efficient":
        return dda_ens(f_scores)
    if strategy == "fast":
        return dda_ns(f_scores)
    raise ValueError("Invalid search strategy")
