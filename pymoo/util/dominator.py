import numpy as np
from scipy.spatial.distance import squareform, pdist

"""
This class enables to compare different solutions according to their domination.
"""


class Dominator:
    def __init__(self):
        pass

    @staticmethod
    def get_constraint_violation(a):
        if a.c is None:
            if a.g is None:
                a.c = 0
            else:
                a.c = np.sum(a.g[a.g > 0])
        return a.c

    @staticmethod
    def is_dominating(a, b):

        if Dominator.get_constraint_violation(a) < Dominator.get_constraint_violation(b):
            return True
        elif Dominator.get_constraint_violation(a) > Dominator.get_constraint_violation(b):
            return False

        all_equal = True
        for i in range(len(a.f)):
            if a.f[i] > b.f[i]:
                return False
            all_equal = all_equal and a.f[i] == b.f[i]
        # if all equal it is not dominating, else it is
        return not all_equal

    @staticmethod
    def get_relation(a, b):

        val = 0

        for i in range(len(a)):

            if a[i] < b[i]:
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1

        return val

    @staticmethod
    def calc_domination_matrix_loop(F, G):
        n = F.shape[0]
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                M[i, j] = Dominator.get_relation(F[i, :], F[j, :])
                M[j, i] = -M[i, j]

        return M

    @staticmethod
    def calc_domination_matrix(F, G):
        smaller = np.any(F[:, np.newaxis] < F, axis=2)
        larger = np.any(F[:, np.newaxis] > F, axis=2)
        M = np.logical_and(smaller, np.logical_not(larger)).astype(np.int) \
            - np.logical_and(larger, np.logical_not(smaller)).astype(np.int)
        return M

