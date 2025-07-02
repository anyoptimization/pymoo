import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.spx import SPX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util import default_random_state
from pymoo.util.display.single import SingleObjectiveOutput


# =========================================================================================================
# Survival
# =========================================================================================================

class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        assert F.shape[1] == 1, "FitnessSurvival can only used for single objective single!"
        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))
        return pop[S[:n_survive]]


# =========================================================================================================
# Implementation
# =========================================================================================================


@default_random_state
def comp_by_cv_and_fitness(pop, P, random_state=None, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True, random_state=random_state)

        # both solutions are feasible just set random
        else:
            S[i] = compare(a, pop[a].F, b, pop[b].F, method='smaller_is_better', return_random_if_equal=True, random_state=random_state)

    return S[:, None].astype(int)


class GA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=FitnessSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=SingleObjectiveOutput(),
                 **kwargs):
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         **kwargs)

        self.termination = DefaultSingleObjectiveTermination()


class BGA(GA):

    def __init__(self,
                 sampling=BinaryRandomSampling(),
                 crossover=SPX(),
                 mutation=BitflipMutation(),
                 **kwargs):
        super().__init__(sampling=sampling, crossover=crossover, mutation=mutation, **kwargs)


parse_doc_string(GA.__init__)
