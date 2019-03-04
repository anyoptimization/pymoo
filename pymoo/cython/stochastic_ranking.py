import numpy as np

from pymoo.rand import random


def stochastic_ranking(F, CV, prob):

    # swap two individuals in the _current population
    def func_swap(A, i, j):
        tmp = A[i]
        A[i] = A[j]
        A[j] = tmp

    # the number of solutions that need to be ranked
    n_solutions = F.shape[0]

    # the number of pairwise comparisons - here we fix it to the number of solutions
    _lambda = n_solutions

    # the final sorting
    index = np.arange(n_solutions)

    for i in range(n_solutions):

        # variable which sets the flag if a swap was performed or not
        swap = False

        for j in range(_lambda - 1):

            _current, _next = index[j], index[j + 1]

            if (CV[_current] == 0 and CV[_next] == 0) or (random.random() < prob):

                if F[_current] > F[_next]:
                    func_swap(index, j, j + 1)
                    swap = True

            else:

                if CV[_current] > CV[_next]:
                    func_swap(index, j, j + 1)
                    swap = True

        if not swap:
            break

    return index