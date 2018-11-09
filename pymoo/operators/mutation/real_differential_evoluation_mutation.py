from pymoo.model.mutation import Mutation
from pymoo.rand import random


class DifferentialEvolutionMutation(Mutation):
    def __init__(self, variant, CR):
        super().__init__(True)
        self.CR = CR
        self.variant = variant

    def _do(self, problem, pop, D=None, **kwargs):

        X = pop.get("X")

        off = D['off']
        _X = off.get("X")

        # do the crossover
        if self.variant == "binomial":
            # uniformly for each individual and each entry
            r = random.random(size=(len(off), problem.n_var)) < self.CR

        X[r] = _X[r]
        return pop.new("X", X)




