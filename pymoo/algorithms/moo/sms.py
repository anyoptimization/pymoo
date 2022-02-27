import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
from pymoo.util.hv import calc_hvc_2d_fast, calc_hvc_looped
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize


# ---------------------------------------------------------------------------------------------------------
# Environmental Survival - Remove the solution with the least HV contribution
# ---------------------------------------------------------------------------------------------------------


class LeastHypervolumeContributionSurvival(Survival):

    def __init__(self, eps=10.0) -> None:
        super().__init__(filter_infeasible=True)
        self.eps = eps

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the number of objectives
        _, n_obj = F.shape

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # set the rank to the current front and initially the hvi to infinity
            pop[front].set("rank", k)

            splitting = len(survivors) + len(front) > n_survive

            if splitting:

                # normalize all the function values for the front
                F = pop[front].get("F")
                F = normalize(F, ideal, nadir)

                I = np.lexsort((-F[:, 1], F[:, 0]))
                F, front = F[I], front[I]

                # define the reference point and shift it a bit - F is already normalized!
                ref_point = np.full(problem.n_obj, 1.0 + self.eps)

                # the solutions select front the splitting front
                S = np.arange(len(front))

                # current front sorted by crowding distance if splitting
                while len(survivors) + len(S) > n_survive:

                    # calculate the hypervolume improvement for each of the individuals (exploit if 2d to be faster)
                    if n_obj == 2:
                        hvi = calc_hvc_2d_fast(F[S], ref_point)
                        # hvi = calc_hvc_2d_slow(F[S], ref_point)
                    else:
                        func = load_function("hv")
                        hvi = calc_hvc_looped(F[S], ref_point, func=func)

                    # the individual to be removed from the current front
                    rem = hvi.argmin()

                    # filter by the individual to be removed
                    S = np.array([s for k, s in enumerate(S) if k != rem])

                front = front[S]

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return pop[survivors]


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament
# ---------------------------------------------------------------------------------------------------------


def cv_and_dom_tournament(pop, P, *args, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            # if one dominates another choose the nds one
            rel = Dominator.get_relation(a_f, b_f)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = np.random.choice([a, b])

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------

class SMSEMOA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=cv_and_dom_tournament),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=LeastHypervolumeContributionSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 normalize=True,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.normalize = normalize

    def _advance(self, infills=None, **kwargs):

        ideal, nadir = None, None

        # estimate ideal and nadir from the current population (more robust then from doing it from merged)
        if self.normalize:
            F = self.pop.get("F")
            ideal, nadir = F.min(axis=0), F.max(axis=0) + 1e-32

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self,
                                    ideal=ideal, nadir=nadir, **kwargs)


parse_doc_string(SMSEMOA.__init__)

