import math

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.decomposition.asf import ASF
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
from pymoo.util.misc import has_feasible, random_permuations
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# Following original code by K. Li https://cola-laboratory.github.io/codes/CTAEA.zip
# =========================================================================================================


def comp_by_cv_dom_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        if pop[a].CV <= 0.0 and pop[b].CV <= 0.0:
            rel = Dominator.get_relation(pop[a].F, pop[b].F)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b
            else:
                S[i] = np.random.choice([a, b])
        elif pop[a].CV <= 0.0:
            S[i] = a
        elif pop[b].CV <= 0.0:
            S[i] = b
        else:
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(int)


class RestrictedMating(TournamentSelection):
    """Restricted mating approach to balance convergence and diversity archives"""

    def _do(self, problem, Hm, n_select, n_parents, **kwargs):
        n_pop = len(Hm) // 2

        _, rank = NonDominatedSorting().do(Hm.get('F'), return_rank=True)

        Pc = (rank[:n_pop] == 0).sum() / len(Hm)
        Pd = (rank[n_pop:] == 0).sum() / len(Hm)

        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure
        n_perms = math.ceil(n_random / n_pop)
        # get random permutations and reshape them
        P = random_permuations(n_perms, n_pop)[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))
        if Pc <= Pd:
            # Choose from DA
            P[::n_parents, :] += n_pop
        pf = np.random.random(n_select)
        P[1::n_parents, :][pf >= Pc] += n_pop

        # compare using tournament function
        S = self.func_comp(Hm, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


class CADASurvival:

    def __init__(self, ref_dirs):
        self.ref_dirs = ref_dirs
        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self._decomposition = ASF()
        self._calc_perpendicular_distance = load_function("calc_perpendicular_distance")

    def do(self, _, pop, da, n_survive=None, **kwargs):
        # Offspring are last of merged population
        off = pop[-n_survive:]
        # Update ideal point
        self.ideal_point = np.min(np.vstack((self.ideal_point, off.get("F"))), axis=0)
        # Update CA
        pop = self._updateCA(pop, n_survive)
        # Update DA
        Hd = Population.merge(da, off)
        da = self._updateDA(pop, Hd, n_survive)
        return pop, da

    def _associate(self, pop):
        """Associate each individual with a F vector and calculate decomposed fitness"""
        F = pop.get("F")
        dist_matrix = self._calc_perpendicular_distance(F - self.ideal_point, self.ref_dirs)
        niche_of_individuals = np.argmin(dist_matrix, axis=1)
        FV = self._decomposition.do(F, weights=self.ref_dirs[niche_of_individuals, :],
                                    ideal_point=self.ideal_point, weight_0=1e-4)
        pop.set("niche", niche_of_individuals)
        pop.set("FV", FV)
        return niche_of_individuals, FV

    def _updateCA(self, pop, n_survive):
        """Update the Convergence archive (CA)"""
        CV = pop.get("CV").flatten()

        Sc = pop[CV == 0]  # ConstraintsAsObjective population
        if len(Sc) == n_survive:  # Exactly n_survive feasible individuals
            F = Sc.get("F")
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            Sc.set('rank', rank)
            self.opt = Sc[fronts[0]]
            return Sc
        elif len(Sc) < n_survive:  # Not enough feasible individuals
            remainder = n_survive - len(Sc)
            # Solve sub-problem CV, tche
            SI = pop[CV > 0]
            f1 = SI.get("CV")
            _, f2 = self._associate(SI)
            sub_F = np.column_stack([f1, f2])
            fronts = NonDominatedSorting().do(sub_F, n_stop_if_ranked=remainder)
            I = []
            for front in fronts:
                if len(I) + len(front) <= remainder:
                    I.extend(front)
                else:
                    n_missing = remainder - len(I)
                    last_front_CV = np.argsort(f1.flatten()[front])
                    I.extend(front[last_front_CV[:n_missing]])
            SI = SI[I]
            S = Population.merge(Sc, SI)
            F = S.get("F")
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            S.set('rank', rank)
            self.opt = S[fronts[0]]
            return S
        else:  # Too many feasible individuals
            F = Sc.get("F")
            # Filter by non-dominated sorting
            fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
            I = np.concatenate(fronts)
            S, rank, F = Sc[I], rank[I], F[I]
            if len(S) > n_survive:
                # Remove individual in most crowded niche and with worst fitness
                niche_of_individuals, FV = self._associate(S)
                index, count = np.unique(niche_of_individuals, return_counts=True)
                survivors = np.full(S.shape[0], True)
                while survivors.sum() > n_survive:
                    crowdest_niches, = np.where(count == count.max())
                    worst_idx = None
                    worst_niche = None
                    worst_fit = -1
                    for crowdest_niche in crowdest_niches:
                        crowdest, = np.where((niche_of_individuals == index[crowdest_niche]) & survivors)
                        niche_worst = crowdest[FV[crowdest].argmax()]
                        dist_to_max_fit = cdist(F[[niche_worst], :], F).flatten()
                        dist_to_max_fit[niche_worst] = np.inf
                        dist_to_max_fit[~survivors] = np.inf
                        min_d_to_max_fit = dist_to_max_fit.min()

                        dist_in_niche = squareform(pdist(F[crowdest]))
                        np.fill_diagonal(dist_in_niche, np.inf)

                        delta_d = dist_in_niche - min_d_to_max_fit
                        min_d_i = np.unravel_index(np.argmin(delta_d, axis=None), dist_in_niche.shape)
                        if (delta_d[min_d_i] < 0) or (
                                delta_d[min_d_i] == 0 and (FV[crowdest[list(min_d_i)]] > niche_worst).any()):
                            min_d_i = list(min_d_i)
                            np.random.shuffle(min_d_i)
                            closest = crowdest[min_d_i]
                            niche_worst = closest[np.argmax(FV[closest])]
                        if (FV[niche_worst] > worst_fit).all():
                            worst_fit = FV[niche_worst]
                            worst_idx = niche_worst
                            worst_niche = crowdest_niche
                    survivors[worst_idx] = False
                    count[worst_niche] -= 1
                S, rank = S[survivors], rank[survivors]
            S.set('rank', rank)
            self.opt = S[rank == 0]
            return S

    def _updateDA(self, pop, Hd, n_survive):
        """Update the Diversity archive (DA)"""
        niche_Hd, FV = self._associate(Hd)
        niche_CA, _ = self._associate(pop)

        itr = 1
        S = []
        while len(S) < n_survive:
            for i in range(n_survive):
                current_ca, = np.where(niche_CA == i)
                if len(current_ca) < itr:
                    for _ in range(itr - len(current_ca)):
                        current_da = np.where(niche_Hd == i)[0]
                        if current_da.size > 0:
                            F = Hd[current_da].get('F')
                            nd = NonDominatedSorting().do(F, only_non_dominated_front=True, n_stop_if_ranked=0)
                            i_best = current_da[nd[np.argmin(FV[current_da[nd]])]]
                            niche_Hd[i_best] = -1
                            if len(S) < n_survive:
                                S.append(i_best)
                        else:
                            break
                if len(S) == n_survive:
                    break
            itr += 1
        return Hd[S]


class CTAEA(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 sampling=FloatRandomSampling(),
                 selection=RestrictedMating(func_comp=comp_by_cv_dom_then_random),
                 crossover=SBX(n_offsprings=1, eta=30, prob=1.0),
                 mutation=PM(prob_var=None, eta=20),
                 eliminate_duplicates=True,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """
        CTAEA

        Parameters
        ----------
        ref_dirs : {ref_dirs}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        """

        self.ref_dirs = ref_dirs
        pop_size = len(ref_dirs)

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = CADASurvival(ref_dirs)

        # Initialize diversity archives
        self.da = None

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=pop_size,
                         output=output,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None and self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.pop, self.da = self.survival.do(self.problem, self.pop, Population(), n_survive=len(self.pop),
                                             algorithm=self)

    def _infill(self):
        Hm = Population.merge(self.pop, self.da)
        return self.mating.do(self.problem, Hm, n_offsprings=self.n_offsprings, algorithm=self)

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."
        pop = Population.merge(self.pop, infills)
        self.pop, self.da = self.survival.do(self.problem, pop, self.da, self.pop_size, algorithm=self)

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt


parse_doc_string(CTAEA.__init__)
