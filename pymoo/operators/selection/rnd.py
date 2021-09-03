import math

import numpy as np

from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations


class RandomSelection(Selection):

    def _do(self, pop, n_select, n_parents, **kwargs):

        # number of random individuals needed
        n_random = n_select * n_parents

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]

        return np.reshape(P, (n_select, n_parents))