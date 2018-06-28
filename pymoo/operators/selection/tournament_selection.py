import math

import numpy as np
from pymoo.model.selection import Selection
from pymoo.rand import random


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
            self.f_comp = self.select_by_min_index

        # attributes that will be set during the optimization
        self.pop = None
        self.perm = None
        self.counter = None

    def _next(self, pop, n_select, n_parents=1, **kwargs):

        l = []
        for i in range(math.ceil(n_select * n_parents * self.pressure / pop.size())):
            l.append(random.perm(size=pop.size()))

        P = np.concatenate(l)[:n_select * n_parents * self.pressure]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        S = self.f_comp(pop, P, **kwargs)
        return np.reshape(S, (n_select, n_parents))

    # select simply by minimum index and assume sorting of the population according selection criteria
    def select_by_min_index(self, pop, P, **kwargs):
        return np.min(P, axis=1)
