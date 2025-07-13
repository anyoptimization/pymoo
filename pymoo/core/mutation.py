from copy import deepcopy

from pymoo.core.operator import Operator
from pymoo.core.variable import Real, get
from pymoo.util import default_random_state


class Mutation(Operator):

    def __init__(self, prob=1.0, prob_var=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prob = Real(prob, bounds=(0.7, 1.0), strict=(0.0, 1.0))
        self.prob_var = Real(prob_var, bounds=(0.0, 0.25), strict=(0.0, 1.0)) if prob_var is not None else None

    @default_random_state
    def do(self, problem, pop, inplace=True, *args, random_state=None, **kwargs):

        # if not inplace copy the population first
        if not inplace:
            pop = deepcopy(pop)

        n_mut = len(pop)

        # get the variables to be mutated
        X = pop.get("X")

        # retrieve the mutation variables
        Xp = self._do(problem, X, *args, random_state=random_state, **kwargs)

        # the likelihood for a mutation on the individuals
        prob = get(self.prob, size=n_mut)
        mut = random_state.random(size=n_mut) <= prob

        # store the mutated individual back to the population
        pop[mut].set("X", Xp[mut])

        return pop

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        return X

    def get_prob_var(self, problem, **kwargs):
        prob_var = self.prob_var if self.prob_var is not None else min(0.5, 1 / problem.n_var)
        return get(prob_var, **kwargs)
