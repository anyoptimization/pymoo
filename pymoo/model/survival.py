from abc import abstractmethod


class Survival:
    """
    The survival process is implemented inheriting from this class, which selects from a population only
    specific individuals to survive.
    """

    def do(self, pop, n_survive, **kwargs):
        """

        The whole population is provided and the number of individuals to survive. If the number of survivers
        is smaller than the populations a survival selection is done. Otherwise, the elements might only
        be sorted by a specific criteria. This depends on the survival implementation.

        Parameters
        ----------
        pop: class
            The population the selected surviving individuals from.
        n_survive: int
            Number of individuals that should survive.
        kwargs: class
            Any additional data that might be needed to choose what individuals survive.

        Returns
        -------
        pop: class
            The population that has survived.

        """
        return self._do(pop, n_survive, **kwargs)

    @abstractmethod
    def _do(self, pop, n_survive, **kwargs):
        pass
