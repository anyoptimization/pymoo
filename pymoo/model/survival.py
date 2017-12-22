from abc import abstractmethod


class Survival:
    """

    The class represents the survival selection during the evolution. Only the fittest can survive.

    """

    def do(self, pop, n_survive, data):
        """

        The whole population is provided and the number of individuals to survive. If the number of survivers
        is smaller than the populations a survival selection is done. Otherwise, the elements might only
        be sorted by a specific criteria. This depends on the survival implementation.

        Parameters
        ----------
        pop: class
            The population.
        n_survive: int
            number of individuals that should survive.
        data: class
            Any additional data that might be needed to choose what individuals survive.

        Returns
        -------
        pop: class
            The population that has survived.

        """
        return self._do(pop, n_survive, data)

    @abstractmethod
    def _do(self, pop, n_survive, data):
        pass
