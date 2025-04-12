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
        fronts: list
            Indices of the individuals in each front.

    References
    ----------
    X. Zhang, Y. Tian, R. Cheng, and Y. Jin,
    An efficient approach to nondominated sorting for evolutionary multiobjective optimization,
    IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.
    """

    assert (strategy in ["sequential", 'binary']), "Invalid search strategy"

    # the shape of the input
    N, M = F.shape

    # do a lexicographic ordering
    I = np.lexsort(F.T[::-1])
    F = F[I]

    # front ranks for each individual
    fronts = []

    for i in range(N):

        if strategy == 'sequential':
            k = sequential_search(F, i, fronts)
        else:
            k = binary_search(F, i, fronts)

        # create empty fronts if necessary
        if k >= len(fronts):
            fronts.append([])

        # append the current individual to a front
        fronts[k].append(i)

    # now map the fronts back to the originally sorting
    ret = []
    for front in fronts:
        ret.append(I[front])

    return ret


def sequential_search(F, i, fronts) -> int:
    """
    Find the front rank for the i-th individual through sequential search.

    Parameters
    ----------
    F: np.ndarray
        the objective values
    i: int
        the index of the individual
    fronts: list
        individuals in each front
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
    Find the front rank for the i-th individual through binary search.

    Parameters
    ----------
    F: np.ndarray
        the objective values
    i: int
        the index of the individual
    fronts: list
        individuals in each front
    """

    num_found_fronts = len(fronts)
    if num_found_fronts == 0:
        return 0

    k_min = 0  # the lower bound for checking
    k_max = num_found_fronts  # the upper bound for checking
    k = floor((k_max + k_min) / 2 + 0.5)  # the front now checked
    current = F[i]
    while True:

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
