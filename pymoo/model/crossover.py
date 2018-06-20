from abc import abstractmethod

import numpy as np


class Crossover:
    """

    This class is the base class for a crossover. In general any crossover can be used.
    It is up to the algorithm composer to take care of not using a crossover for real values in a binary problem.
    Each crossover needs to define the number of parents that are needed and also the number of children
    that are composed.

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

        parents: matrix
            The parents as a matrix. Each row is a parent and the columns are equal to the parameter length.

        data:
            Any additional data that might be necessary to perform the crossover. E.g. constants of an algorithm.

        Returns
        -------
        children: matrix
            The children as a matrix. n_children rows and the number of columns is equal to the variable
            length of the problem.

        """
        n_var = parents.shape[1]

        #if n_var != problem.n_var:
        #    raise ValueError('Exception during crossover: Variable length is not equal to the defined one in problem.')

        #if self.n_parents != parents.shape[0]:
        #    raise ValueError('Exception during crossover: Number of parents differs from defined at crossover.')

        children = np.full((self.n_children, n_var), np.inf)
        return self._do(problem, parents, children,  **kwargs)


    @abstractmethod
    def _do(self, problem, parents, children,  **kwargs):
        pass
