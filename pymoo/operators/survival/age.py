from pymoo.model.population import Population
from pymoo.model.survival import Survival


class AgeBasedSurvival(Survival):

    def __init__(self, survival, n_max_age) -> None:
        super().__init__()
        self.survival = survival
        self.n_max_age = n_max_age

    def _do(self, problem, pop, n_survive, algorithm=None, **kwargs):

        if algorithm is None:
            raise Exception("Algorithm object needs to be passed to determine the current generation!")

        # get the generations of the population
        gen = pop.get("n_gen")

        # by default will with the current generation if unknown (should not happen)
        for k in range(len(gen)):
            if gen[k] is None:
                gen[k] = algorithm.n_gen

        # get the age of each individual
        age = algorithm.n_gen - gen

        # define what individuals are too old and which are young enough
        too_old = age > self.n_max_age
        young_enough = ~too_old

        # initialize the survivors
        survivors = Population()

        # do the survival with individuals being young enough
        if young_enough.sum() > 0:
            survivors = self.survival.do(problem, pop[young_enough], n_survive, algorithm=algorithm, **kwargs)

        n_remaining = n_survive - len(survivors)

        # if really necessary fill up with individuals which are actually too old
        if n_remaining > 0:
            fill_up = pop[too_old]
            fill_up = fill_up[fill_up.get("n_gen").argsort()]

            survivors = Population.merge(survivors, fill_up[:n_remaining])

        return survivors
