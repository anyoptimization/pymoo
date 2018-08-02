import math

import numpy as np

from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations


class TournamentSelection(Selection):
    """
      The Tournament selection is used to simulated a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, f_comp=None, pressure=2):
        """

        Parameters
        ----------
        f_comp: func
            The function to compare two individuals. It has the shape: comp(pop, indices, data) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterium and only indices are compared.

        pressure: int
            The selection pressure to be applied. Default it is a binary tournament.
        """

        # selection pressure to be applied
        self.pressure = pressure

        self.f_comp = f_comp
        if self.f_comp is None:
            raise Exception("Please provide the comparing function for the tournament selection!")

        # attributes that will be set during the optimization
        self.pop = None
        self.perm = None
        self.counter = None

    def _next(self, pop, n_select, n_parents=1, **kwargs):

        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / pop.size())

        # get random permutations and reshape them
        P = random_permuations(n_perms, pop.size())[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.f_comp(pop, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))
