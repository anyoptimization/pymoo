import numpy as np

from pymoo.model.population import Population


class Crossover:
    """
    The crossover combines parents to offsprings. Some crossover are problem specific and use additional information.
    This class must be inherited from to provide a crossover method to an algorithm.
    """

    def __init__(self, n_parents, n_offsprings, prob=0.9):
        self.prob = prob
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

    def do(self, problem, pop, parents, **kwargs):
        """

        This method executes the crossover on the parents. This class wraps the implementation of the class
        that implements the crossover.

        Parameters
        ----------
        problem: class
            The problem to be solved. Provides information such as lower and upper bounds or feasibility
            conditions for custom crossovers.

        pop : Population
            The population as an object

        parents: numpy.array
            The select parents of the population for the crossover

        kwargs : dict
            Any additional data that might be necessary to perform the crossover. E.g. constants of an algorithm.

        Returns
        -------
        offsprings : Population
            The off as a matrix. n_children rows and the number of columns is equal to the variable
            length of the problem.

        """

        if self.n_parents != parents.shape[1]:
            raise ValueError('Exception during crossover: Number of parents differs from defined at crossover.')

        # get the design space matrix form the population and parents
        X = pop.get("X")[parents.T].copy()

        # now apply the crossover probability
        do_crossover = np.random.random(len(parents)) < self.prob

        # execute the crossover
        _X = self._do(problem, X, **kwargs)

        X[:, do_crossover, :] = _X[:, do_crossover, :]

        # flatten the array to become a 2d-array
        X = X.reshape(-1, X.shape[-1])

        # create a population object
        off = pop.new("X", X)

        return off
