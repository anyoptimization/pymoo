import math

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga3 import comp_by_cv_then_random
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.function_loader import load_function
from pymoo.util.misc import has_feasible, random_permuations
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class RestrictedMating(TournamentSelection):

    def _do(self, Hm, n_select, n_parents=1, **kwargs):
        algorithm = kwargs['algorithm']

        n_pop = len(Hm) // 2

        _, rank = NonDominatedSorting().do(Hm.get('F'), return_rank=True)

        Pc = (rank[:n_pop] == 0).sum()/len(Hm)
        Pd = (rank[n_pop:] == 0).sum()/len(Hm)
        PC = len(algorithm.opt) / n_pop

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
        P[1::2, 1][pf >= PC] += n_pop

        # compare using tournament function
        S = self.f_comp(Hm, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


class CTAEA(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 sampling=FloatRandomSampling(),
                 selection=RestrictedMating(func_comp=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(n_offsprings=1, eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=True,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

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

        kwargs['individual'] = Individual(rank=np.inf)

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
                         display=display,
                         **kwargs)

    def _initialize(self):
        # Prepare diversity archives
        da = Population(0, individual=self.individual)

        # create the initial population
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop, da = self.survival.do(self.problem, pop, da, len(pop), algorithm=self)

        self.pop = pop
        self.da = da

    def _solve(self, problem):

        if self.ref_dirs is not None and self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))

        return super()._solve(problem)

    def _next(self):

        # do the mating using the total population
        Hm = self.pop.merge(self.da)
        self.off = self.mating.do(self.problem, Hm, n_offsprings=self.n_offsprings, algorithm=self)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = self.pop.merge(self.off)

        # the do survival selection
        self.pop, self.da = self.survival.do(self.problem, self.pop, self.da, self.pop_size, algorithm=self)

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt


class CADASurvival:

    def __init__(self, ref_dirs):
        self.ref_dirs = ref_dirs
        self.opt = None
        self._decomposition = get_decomposition('tchebi')
        self._calc_perpendicular_distance = load_function("calc_perpendicular_distance")

    def do(self, problem, pop, da, n_survive, **kwargs):
        off = pop[-n_survive:]
        pop = self._updateCA(pop, n_survive)
        Hd = da.merge(off)
        da = self._updateDA(pop, Hd, n_survive)
        return pop, da

    def _association(self, F):
        dist_matrix = self._calc_perpendicular_distance(F, self.ref_dirs)
        niche_of_individuals = np.argmin(dist_matrix, axis=1)
        return niche_of_individuals

    def _get_decomposition(self, F):
        niche_of_individuals = self._association(F)
        ideal_point = np.min(F, axis=0)
        return self._decomposition.do(
            F, weights=self.ref_dirs[niche_of_individuals, :],
            ideal_point=ideal_point)

    def _updateCA(self, pop, n_survive):
        CV = pop.get("CV").flatten()

        Sc = pop[CV == 0]
        if len(Sc) == n_survive:
            F = Sc.get("F")
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            Sc.set('rank', rank)
            self.opt = Sc[fronts[0]]
            return Sc
        elif len(Sc) < n_survive:
            remainder = n_survive-len(Sc)
            # Solve sub-problem CV, tche
            SI = pop[CV > 0]
            f1 = SI.get("CV")
            F = SI.get("F")
            f2 = self._get_decomposition(F)
            sub_F = np.column_stack([f1, f2])
            fronts, rank = NonDominatedSorting().do(sub_F, return_rank=True, n_stop_if_ranked=remainder)
            I = np.concatenate(fronts)
            SI = SI[I]
            if len(SI) > remainder:
                SI = SI[np.argsort(f1.flatten()[I])[:remainder]]
            S = Sc.merge(SI)
            F = S.get("F")
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            S.set('rank', rank)
            self.opt = S[fronts[0]]
            return S
        else:  # len(Sc) > n_survive
            F = Sc.get("F")
            fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
            I = np.concatenate(fronts)
            S, rank, F = Sc[I], rank[I], F[I]

            if len(S) > n_survive:
                self.nadir_point = np.max(F, axis=0)
                self.ideal_point = np.min(F, axis=0)
                nF = (F - self.ideal_point) / (self.nadir_point - self.ideal_point)
                niche_of_individuals = self._association(nF)
                index, count = np.unique(niche_of_individuals, return_counts=True)
                survivors = np.full(S.shape[0], True)
                while survivors.sum() > n_survive:
                    crowdest_niche = np.argmax(count)
                    crowdest = np.where((niche_of_individuals == index[crowdest_niche]) & survivors)[0]
                    crowdest_F = F[crowdest, :]
                    dist = pdist(crowdest_F)
                    sdist = squareform(dist)
                    sdist[sdist == 0] == np.inf
                    min_d_i = np.unravel_index(np.argmin(sdist, axis=None), sdist.shape)
                    St_F = crowdest_F[min_d_i, :]
                    St_FV = self._get_decomposition(St_F)
                    survivors[crowdest[min_d_i[np.argmax(St_FV)]]] = False
                    count[crowdest_niche] -= 1
                S, rank = S[survivors], rank[survivors]
            S.set('rank', rank)
            self.opt = S[rank == 0]
            return S

    def _updateDA(self, pop, Hd, n_survive):
        niche_Hd = self._association(Hd.get('F'))
        niche_CA = self._association(pop.get('F'))

        itr = 1
        S = []
        while len(S) < n_survive:
            for i in range(n_survive):
                current_ca = np.where(niche_CA == i)
                if len(current_ca) < itr:
                    for _ in range(itr - len(current_ca)):
                        current_da = np.where(niche_Hd == i)[0]
                        if current_da.size > 0:
                            F = Hd[current_da].get('F')
                            nd = NonDominatedSorting().do(F, only_non_dominated_front=True, n_stop_if_ranked=0)
                            FV = self._get_decomposition(F[nd])
                            i_best = current_da[nd[np.argmin(FV)]]
                            niche_Hd[i_best] = -1
                            if len(S) < n_survive:
                                S.append(i_best)
                        else:
                            break
                if len(S) == n_survive:
                    break
            itr += 1
        return Hd[S]
