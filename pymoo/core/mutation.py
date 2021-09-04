class Mutation:

    def __init__(self) -> None:
        super().__init__()
        self.algorithm = None
        self.problem = None

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
        Xp : Population
            The mutated population.

        """

        Xp = self._do(problem, pop.get("X"), **kwargs)
        return pop.new("X", Xp)

    def _do(self, problem, X, **kwargs):
        pass
