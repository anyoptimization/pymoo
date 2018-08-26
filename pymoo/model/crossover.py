from abc import abstractmethod

import numpy as np


class Crossover:
    """
    The crossover combines parents to offsprings. Some crossover are problem specific and use additional information.
    This class must be inherited from to provide a crossover method to an algorithm.
    """

    def __init__(self, n_parents, n_children):
        self.n_parents = n_parents
        self.n_children = n_children

    def do(self, problem, parents, **kwargs):
        """

        This method executes the crossover on the parents. This class wraps the implementation of the class
        that implements the crossover.

        Parameters
        ----------
        problem: class
            The problem to be solved. Provides information such as lower and upper bounds or feasibility
            conditions for custom crossovers.

        parents: numpy.ndarray
            The parents as a matrix. Each row is a parent and the columns are equal to the parameter length.

        kwargs : dict:
            Any additional data that might be necessary to perform the crossover. E.g. constants of an algorithm.

        Returns
        -------
        children: np.ndarray
            The children as a matrix. n_children rows and the number of columns is equal to the variable
            length of the problem.

        """
        n_matings = parents.shape[1]
        n_var = parents.shape[2]

        if n_var != problem.n_var:
            raise ValueError('Exception during crossover: Variable length is not equal to the defined one in problem.')

        if self.n_parents != parents.shape[0]:
            raise ValueError('Exception during crossover: Number of parents differs from defined at crossover.')

        children = np.full((n_matings * self.n_children, n_var), np.inf)
        self._do(problem, parents, children, **kwargs)
        return children

    @abstractmethod
    def _do(self, problem, parents, children, **kwargs):
        n_children = 0
        for k in range(parents.shape[1]):

            _children = np.full((self.n_children, problem.n_var), np.inf)

            self._mating(problem, parents[:, k, :], _children, **kwargs)

            for i in range(_children.shape[0]):
                children[n_children + i, :] = _children[i, :]

            n_children += self.n_children

    @abstractmethod
    def _mating(self, problem, parents, children, **kwargs):
        pass
