from abc import abstractmethod

import numpy as np

from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling
from pymoo.core.parameters import get_params, flatten
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Real, Integer, Binary
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.integer_from_float_operator import IntegerFromFloatCrossover, IntegerFromFloatMutation
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.selection.tournament import TournamentSelection, compare


class ParameterControl:

    def __init__(self, obj) -> None:
        super().__init__()

        self.data = None

        params = flatten({"ParameterControl": get_params(obj)})
        self.params = params

        # print("PARAMETER CONTROL:", list(self.params))

    def do(self, N, set_to_params=True):
        vals = self._do(N)
        if set_to_params:
            if vals is not None:
                for k, v in vals.items():
                    self.params[k].set(v)
        return vals

    @abstractmethod
    def _do(self, N):
        pass

    def tell(self, **kwargs):
        self.data = dict(kwargs)

    def advance(self, infills=None):
        for k, v in self.params.items():
            assert len(v.get()) == len(
                infills), "Make sure that the infills and parameters asked for have the same size."
            infills.set(k, v.get())


class NoParameterControl(ParameterControl):

    def __init__(self, _) -> None:
        super().__init__(None)

    def _do(self, N):
        return {}


class RandomParameterControl(ParameterControl):

    def _do(self, N):
        return {key: value.sample(N) for key, value in self.params.items()}


class EvolutionaryParameterControl(ParameterControl):

    def __init__(self, obj) -> None:
        super().__init__(obj)
        self.eps = 0.05

    def _do(self, N):
        params = self.params
        pop = self.data.get("pop")

        # make sure that for each parameter a value exists - if not simply set it randomly
        for name, param in params.items():
            is_none = np.where(pop.get(name) == None)[0]
            if len(is_none) > 0:
                pop[is_none].set(name, param.sample(len(is_none)))

        crossover = {
            Binary: UX(),
            Real: SBX(prob=0.9, prob_var=0.5, prob_exchange=1.0),
            Integer: IntegerFromFloatCrossover(SBX),
            Choice: UX(),
        }

        mutation = {
            Binary: BFM(),
            Real: PM(prob=1.0),
            Integer: IntegerFromFloatMutation(PM),
            Choice: ChoiceRandomMutation(),
        }

        mating = MixedVariableMating(
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=NoDuplicateElimination()
        )

        parents = AgeBasedTournamentSelection().do(None, pop, N, n_parents=2, to_pop=False)

        problem = Problem(vars=params)
        X = [{key: ind.get(key) for key in params} for ind in pop]
        parent_pop = Population.new(X=X)
        off = mating.do(problem, parent_pop, N, parents=parents)

        # with probability of eps -> use a random solution
        # rnd = np.random.random(len(off)) < self.eps
        # off[rnd] = MixedVariableSampling().do(problem, rnd.sum())

        Xp = off.get("X")
        ret = {param: np.array([Xp[i][param] for i in range(len(Xp))]) for param in params}

        return ret


class AgeBasedTournamentSelection(TournamentSelection):

    def __init__(self, pressure=2):
        super().__init__(age_binary_tournament, pressure)


def age_binary_tournament(pop, P, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_gen, b_gen = pop[a].get("n_gen"), pop[b].get("n_gen")
        S[i] = compare(a, a_gen, b, b_gen, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)
