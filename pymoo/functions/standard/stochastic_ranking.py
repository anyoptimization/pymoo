"""Standard Python implementation of stochastic ranking."""

import numpy as np

from pymoo.util import default_random_state


def swap(M, a, b):
    """Swap two elements in array M.

    Args:
        M: Array to modify.
        a: First index.
        b: Second index.
    """
    M[a], M[b] = M[b], M[a]


@default_random_state
def stochastic_ranking(f, phi, pr, I=None, random_state=None):  # noqa: E741
    """Stochastic ranking algorithm.

    Args:
        f: Objective values.
        phi: Constraint violation values.
        pr: Probability of comparison.
        I: Indices array.
        random_state: Random state.

    Returns:
        Sorted indices.
    """
    _lambda = len(f)

    if I is None:  # noqa: E741
        I = np.arange(_lambda)  # noqa: E741

    for i in range(_lambda):
        at_least_one_swap = False

        for j in range(_lambda - 1):
            u = random_state.random()

            if u < pr or (phi[I[j]] == 0 and phi[I[j + 1]] == 0):
                if f[I[j]] > f[I[j + 1]]:
                    swap(I, j, j + 1)
                    at_least_one_swap = True
            else:
                if phi[I[j]] > phi[I[j + 1]]:
                    swap(I, j, j + 1)
                    at_least_one_swap = True

        if not at_least_one_swap:
            break

    return I
