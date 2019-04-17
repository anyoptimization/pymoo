import time

import numpy as np
from numba import njit
from pymoo.cython.non_dominated_sorting_cython import fast_non_dominated_sort as fast_fast_non_dominated_sort

from pymoo.util.dominator import Dominator


@njit()
def fast_non_dominated_sort_numba(M):

    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = [0 for _ in range(n)]

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = []

    # storage for the number of solutions dominated this one
    n_dominated = [0 for _ in range(n)]

    current_front = []

    for i in range(n):

        _is_dominating = []

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                _is_dominating.append(j)
                n_dominated[j] += 1
            elif rel == -1:
                _is_dominating.append(i)
                n_dominated[i] += 1

        is_dominating.append(_is_dominating)

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1

                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts



def fast_non_dominated_sort(F, **kwargs):
    M = Dominator.calc_domination_matrix(F)

    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=np.int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts


def non_dominated_sort_naive(F, **kwargs):
    M = Dominator.calc_domination_matrix(F)

    fronts = []
    remaining = set(range(M.shape[0]))

    while len(remaining) > 0:

        front = []

        for i in remaining:

            is_dominated = False
            dominating = set()

            for j in front:
                rel = M[i, j]
                if rel == 1:
                    dominating.add(j)
                elif rel == -1:
                    is_dominated = True
                    break

            if is_dominated:
                continue
            else:
                front = [x for x in front if x not in dominating]
                front.append(i)

        [remaining.remove(e) for e in front]
        fronts.append(front)

    return fronts


if __name__ == "__main__":
    X = np.random.random((1000, 3))

    start = time.time()
    fast_non_dominated_sort(X)
    end = time.time()
    print("Elapsed (Pure Python) = %s" % (end - start))

    start = time.time()
    fast_fast_non_dominated_sort(X)
    end = time.time()
    print("Elapsed (Cython) = %s" % (end - start))

    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.time()
    print(fast_non_dominated_sort_numba.inspect_types())
    fast_non_dominated_sort_numba(Dominator.calc_domination_matrix(X))
    end = time.time()
    print("Elapsed (Numba with compilation) = %s" % (end - start))

    # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    start = time.time()
    fast_non_dominated_sort_numba(Dominator.calc_domination_matrix(X))
    end = time.time()
    print("Elapsed (Numba after compilation) = %s" % (end - start))
