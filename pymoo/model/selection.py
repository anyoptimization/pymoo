from abc import abstractmethod


class Selection:
    """

    This class represents the selection of individuals for the mating.

    """

    @abstractmethod
    def set_population(self, pop, data):
        """

        Set the population to be selected from.

        Parameters
        ----------
        pop: class
            An object that stores the current population to be selected from.
        data: class
            Any other additional data that might be needed for the selection procedure.

        """
        pass

    @abstractmethod
    def next(self, n_select):
        """

        Choose from the population new individuals to be selected.

        Parameters
        ----------
        n_select: int
            How many individuals that should be selected.

        Returns
        -------

        select: vector
            A vector that contains the result indices of selected individuals.

        """

        pass
