import numpy as np

"""
This class enables to compare different solutions according to their domination.
"""


class Dominator:
    def __init__(self):
        pass

    @staticmethod
    def get_constraint_violation(a):
        if a.c is None:
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

        # calculate sum of constraint violation if not done before
        Dominator.get_constraint_violation(a)
        Dominator.get_constraint_violation(b)

        # check out the constraint violation first
        if a.c < b.c:
            return 1
        elif b.c < a.c:
            return -1
        else:

            # iterate through the objectives to check the relation
            val = 0

            for i in range(len(a.f)):

                if a.f[i] < b.f[i]:
                    # indifferent because once better and once worse
                    if val == -1:
                        return 0
                    val = 1
                elif b.f[i] < a.f[i]:
                    # indifferent because once better and once worse
                    if val == 1:
                        return 0
                    val = -1

        return val
