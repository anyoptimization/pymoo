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
        self.data = None

    def set_population(self, pop, data):
        """

        Parameters
        ----------
        pop: class
            The population to be selected from.
        data: class
            Any additional data that might be needed for the selection.

        """
        self.pop = pop
        self.data = data
        self.perm = random.perm(self.pop.size())
        self.counter = 0

    def next(self, n_selected):
        """

        Parameters
        ----------
        n_selected: int
            Number of individuals to be selected.

        Returns
        -------
        v: vector
            Selected indices of individuals as a integer vector.

        """

        selected = np.zeros(n_selected, dtype=np.int)

        for i in range(n_selected):

            if self.counter + self.pressure >= self.pop.size():
                self.perm = random.perm(self.pop.size())
                self.counter = 0
            selected[i] = self.f_comp(self.pop, self.perm[self.counter:self.counter + self.pressure], self.data)
            self.counter = self.counter + self.pressure

        return selected

    # select simply by minimum index and assume sorting of the population according selection criteria
    def select_by_min_index(self, pop, indices, data):
        return np.min(indices)
