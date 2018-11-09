from abc import abstractmethod


class Mutation:

    def __init__(self, vectorized) -> None:
        super().__init__()
        self.vectorized = vectorized

    def do(self, problem, pop, **kwargs):
        """

        Mutate variables in a genetic way.

        Parameters
        ----------
        problem : class
            The problem instance - specific information such as variable bounds might be needed.
        pop : Population
            A population object

        Returns
        -------
        Y : Population
            The mutated population.

        """

        return self._do(problem, pop, **kwargs)



    @abstractmethod
    def _do(self, problem, pop, **kwargs):
        pass
