from abc import abstractmethod


class Selection:

    def __init__(self) -> None:
        """
        This class is used to select parents for the mating or other evolutionary operators.
        Several strategies can be used to increase the selection pressure.
        """
        super().__init__()

    def do(self, pop, n_select, n_parents=2, **kwargs):
        """
        Choose from the population new individuals to be selected.

        Parameters
        ----------
        pop : :class:`~pymoo.model.population.Population`
            The population which should be selected from. Some criteria from the design or objective space
            might be used for the selection. Therefore, only the number of individual might be not enough.

        n_select : int
            Number of individuals to select.

        n_parents : int
            Number of parents needed to create an offspring.

        Returns
        -------
        I : numpy.array
            Indices of selected individuals.

        """

        return self._do(pop, n_select, n_parents, **kwargs)

    @abstractmethod
    def _do(self, pop, n_select, n_parents, **kwargs):
        pass