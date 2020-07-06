from math import floor

import numpy as np

from pymoo.util.dominator import Dominator


def efficient_non_dominated_sort(F, strategy="sequential"):
    """
    Efficient Non-dominated Sorting (ENS)
    Parameters
    ----------
    F: numpy.ndarray
        objective values for each individual.
    strategy: str
        search strategy, can be "sequential" or "binary".
    Returns
    -------
        indices of the individuals in each front.

    References
    ----------
    X. Zhang, Y. Tian, R. Cheng, and Y. Jin,
    An efficient approach to nondominated sorting for evolutionary multiobjective optimization,
    IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.
    """
    assert (strategy in ["sequential", 'binary']), "Invalid search strategy"
    N, M = F.shape
    # sort the rows in F
    indices = sort_rows(F)
    F = F[indices]
    # front ranks for each individual
    fronts = []  # front with sorted indices
    _fronts = []  # real fronts
    for i in range(N):
        if strategy == 'sequential':
            k = sequential_search(F, i, fronts)
        else:
            k = binary_search(F, i, fronts)
        if k >= len(fronts):
            fronts.append([])
            _fronts.append([])
        fronts[k].append(i)
        _fronts[k].append(indices[i])
    return _fronts


def sequential_search(F, i, fronts) -> int:
    """
    Find the front rank for the i-th individual through sequential search
    Parameters
    ----------
    F: the objective values
    i: the index of the individual
    fronts: individuals in each front
    """
    num_found_fronts = len(fronts)
    k = 0  # the front now checked
    current = F[i]
    while True:
        if num_found_fronts == 0:
            return 0
        # solutions in the k-th front, examine in reverse order
        fk_indices = fronts[k]
        solutions = F[fk_indices[::-1]]
        non_dominated = True
        for f in solutions:
            relation = Dominator.get_relation(current, f)
            if relation == -1:
                non_dominated = False
                break
        if non_dominated:
            return k
        else:
            k += 1
            if k >= num_found_fronts:
                # move the individual to a new front
                return num_found_fronts


def binary_search(F, i, fronts):
    """
    Find the front rank for the i-th individual through binary search
    Parameters
    ----------
    F: the objective values
    i: the index of the individual
    fronts: individuals in each front
    """
    num_found_fronts = len(fronts)
    k_min = 0  # the lower bound for checking
    k_max = num_found_fronts  # the upper bound for checking
    k = floor((k_max + k_min) / 2 + 0.5)  # the front now checked
    current = F[i]
    while True:
        if num_found_fronts == 0:
            return 0
        # solutions in the k-th front, examine in reverse order
        fk_indices = fronts[k - 1]
        solutions = F[fk_indices[::-1]]
        non_dominated = True
        for f in solutions:
            relation = Dominator.get_relation(current, f)
            if relation == -1:
                non_dominated = False
                break
        # binary search
        if non_dominated:
            if k == k_min + 1:
                return k - 1
            else:
                k_max = k
                k = floor((k_max + k_min) / 2 + 0.5)
        else:
            k_min = k
            if k_max == k_min + 1 and k_max < num_found_fronts:
                return k_max - 1
            elif k_min == num_found_fronts:
                return num_found_fronts
            else:
                k = floor((k_max + k_min) / 2 + 0.5)


def sort_rows(array, order='asc'):
    """
    Sort the rows of an array in ascending order.
    The algorithm will try to use the first column to sort the rows of the given array. If ties occur, it will use the
    second column, and so on.
    Parameters
    ----------
    array: numpy.ndarray
        array to be sorted
    order: str
        sort order, can be 'asc' or 'desc'
    Returns
    -------
    the indices of the rows in the sorted array.
    """
    assert (order in ['asc', 'desc']), "Invalid sort order!"
    ix = np.lexsort(array.T[::-1])
    return ix if order == 'asc' else ix[::-1]
