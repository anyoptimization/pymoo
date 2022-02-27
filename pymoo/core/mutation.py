import numpy as np

from pymoo.core.operator import Operator
from pymoo.core.variable import Real, get


class Mutation(Operator):

    def __init__(self, prob=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prob = Real(prob, bounds=(0.7, 1.0), strict=(0.0, 1.0))

    def do(self, problem, pop, **kwargs):
        Xp = self._do(problem, pop.get("X"), **kwargs)

        prob = get(self.prob, size=len(pop))
        mut = np.random.random() <= prob

        pop[mut].set("X", Xp[mut])
        return pop

    def _do(self, problem, X, **kwargs):
        pass


class VariableWiseMutation(Mutation):

    def __init__(self, prob_var=None, **kwargs):
        super().__init__(**kwargs)
        self.prob_var = prob_var

    def get_prob_var(self, problem, **kwargs):
        prob_var = get(self.prob_var)
        if prob_var is None:
            prob_var = 1 / problem.n_var
        return get(prob_var, **kwargs)
