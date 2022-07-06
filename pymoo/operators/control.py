import math
from abc import abstractmethod

import numpy as np

from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.infill import InfillCriterion
from pymoo.core.mixed import MixedVariableMating
from pymoo.core.parameters import get_params, flatten
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Real, Integer, Binary
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.repair.rounding import RoundingRepair
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

        selection = AgeBasedTournamentSelection()

        crossover = {
            Binary: UX(),
            Real: SBX(),
            Integer: SBX(vtype=float, repair=RoundingRepair()),
            Choice: UX(),
        }

        mutation = {
            Binary: BFM(),
            Real: PM(),
            Integer: PM(vtype=float, repair=RoundingRepair()),
            Choice: ChoiceRandomMutation(),
        }

        mating = MixedVariableMating(
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=NoDuplicateElimination()
        )

        problem = Problem(vars=params)

        parents = selection(problem, pop, N, n_parents=2)
        parents = [[Individual(X={key: parent.get(key) for key in params}) for parent in mating] for mating in parents]

        off = mating(problem, parents, N, parents=True)

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


class ParameterControlMating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 control=NoParameterControl,
                 **kwargs):
        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.control = control(self)

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # do the parameter control for the mating
        control = self.control
        control.tell(pop=pop)
        control.do(n_matings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection.do(problem, pop, n_matings, n_parents=self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(problem, parents, **kwargs)

        # now we have to consider during parameter control that a crossover can produce multiple offsprings
        for name, param in control.params.items():
            param.set(np.repeat(param.get(), self.crossover.n_offsprings))

        # do the mutation on the offsprings created through crossover
        off = self.mutation(problem, off, **kwargs)

        # finally attach the parameters back to the offsprings
        control.advance(off)

        return off
