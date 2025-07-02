import math

import numpy as np

from pymoo.core.selection import Selection
from pymoo.util.misc import random_permutations
from pymoo.util import default_random_state


class TournamentSelection(Selection):
    """
      The Tournament selection is used to simulate a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, func_comp=None, pressure=2, **kwargs):
        """

        Parameters
        ----------
        func_comp: func
            The function to compare two individuals. It has the shape: comp(pop, indices) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterion and only indices are compared.

        pressure: int
            The selection pressure to be applied. Default it is a binary tournament.
        """

        super().__init__(**kwargs)

        # selection pressure to be applied
        self.pressure = pressure

        self.func_comp = func_comp
        if self.func_comp is None:
            raise Exception("Please provide the comparing function for the tournament selection!")

    def _do(self, _, pop, n_select, n_parents=1, random_state=None, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permutations(n_perms, len(pop), random_state=random_state)[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.func_comp(pop, P, random_state=random_state, **kwargs)

        return np.reshape(S, (n_select, n_parents))


@default_random_state
def compare(a, a_val, b, b_val, method, return_random_if_equal=False, random_state=None):
    if method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return random_state.choice([a, b])
            else:
                return None
    elif method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return random_state.choice([a, b])
            else:
                return None
    else:
        raise Exception("Unknown method.")
