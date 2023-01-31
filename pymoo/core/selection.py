from abc import abstractmethod

import numpy as np

from pymoo.core.operator import Operator


class Selection(Operator):

    def __init__(self, **kwargs) -> None:
        """
        This class is used to select parents for the mating or other evolutionary operators.
        Several strategies can be used to increase the selection pressure.
        """
        super().__init__(**kwargs)

    def do(self, problem, pop, n_select, n_parents, to_pop=True, **kwargs):
        """
        Choose from the population new individuals to be selected.

        Parameters
        ----------


        problem: class
            The problem to be solved. Provides information such as lower and upper bounds or feasibility
            conditions for custom crossovers.

        pop : :class:`~pymoo.core.population.Population`
            The population which should be selected from. Some criteria from the design or objective space
            might be used for the selection. Therefore, only the number of individual might be not enough.

        n_select : int
            Number of individuals to select.

        n_parents : int
            Number of parents needed to create an offspring.

        to_pop : bool
            Whether IF(!) the implementation returns only indices, it should be converted to individuals.

        Returns
        -------
        parents : list
            List of parents to be used in the crossover

        """

        ret = self._do(problem, pop, n_select, n_parents, **kwargs)

        # if some selections return indices they are used to create the individual list
        if to_pop and isinstance(ret, np.ndarray) and np.issubdtype(ret.dtype, np.integer):
            ret = pop[ret]

        return ret

    @abstractmethod
    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        pass


