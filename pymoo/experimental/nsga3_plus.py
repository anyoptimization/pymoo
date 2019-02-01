import numpy as np
from pymoo.operators.selection.random_selection import RandomSelection

from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover

from pymoo.operators.selection.tournament_selection import TournamentSelection

from pymoo.model.crossover import Crossover

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga3 import get_extreme_points_c, get_nadir_point, associate_to_niches, calc_niche_count, \
    niching, comp_by_cv_then_random
from pymoo.model.individual import Individual
from pymoo.model.selection import Selection
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.misc import cdist
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class NSGA3Plus(GeneticAlgorithm):

    def __init__(self, ref_dirs, **kwargs):
        self.ref_dirs = ref_dirs
        kwargs['individual'] = Individual(rank=np.inf, niche=-1, dist_to_niche=np.inf)
        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', LatinHypercubeSampling(iterations=100))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=30))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=20))
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        #set_if_none(kwargs, 'selection', RandomSelection())
        set_if_none(kwargs, 'survival', ReferenceDirectionSurvivalPlus(ref_dirs))
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective
        self.n_offsprings = len(ref_dirs)

    def _solve(self, problem, termination):
        if self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))

        return super()._solve(problem, termination)

    def _next2(self, pop):

        # what individuals are feasible
        _feasible = pop.get("feasible")

        infeasible = np.where(np.logical_not(_feasible))[0]
        feasible = np.where(_feasible)[0]

        if len(infeasible) == 0:

            self.selection = TournamentSelection(func_comp=comp_by_cv_then_random)
            self.crossover = SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=30)
            off_feasbible = self._mating(pop[feasible])

            self.off = off_feasbible

        else:

            prob_feasible, prob_infeasible, prob_mixed = 0.33, 0.33, 0.33

            self.selection = RandomSelection()
            self.crossover = SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=30)
            off_feasbible = self._mating(pop[feasible])

            self.selection = RandomSelection()
            self.crossover = DifferentialEvolutionCrossover(weight=0.75)
            off_infeasbible = self._mating(pop[infeasible])

            # bring back to bonds because of DE
            X = off_infeasbible.get("X")
            xl = np.repeat(self.problem.xl[None, :], X.shape[0], axis=0)
            xu = np.repeat(self.problem.xu[None, :], X.shape[0], axis=0)
            X[X < xl] = (xl + (xl - X))[X < xl]
            X[X > xu] = (xu - (X - xu))[X > xu]
            off_infeasbible.set("X", X)

            self.selection = MySelection(5)
            self.crossover = SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=30)
            off_mixed = self._mating(pop)

            r = random.random(self.n_offsprings)
            self.off = off_feasbible

            b = np.logical_and(prob_feasible <= r, r < prob_feasible + prob_infeasible)
            self.off[b] = off_infeasbible[b]

            b = r >= prob_feasible + prob_infeasible
            self.off[b] = off_mixed[b]

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        pop = pop.merge(self.off)

        # the do survival selection
        pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        return pop


class MySelection(Selection):

    def __init__(self, n_neighbors) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors

    def _do(self, pop, n_select, n_parents=1, **kwargs):
        X, feasible = pop.get("X", "feasible")

        infeasible = np.where(np.logical_not(feasible))[0]
        feasible = np.where(feasible)[0]

        D = cdist(X[feasible], X[infeasible])

        S = []

        while len(S) < n_select:
            _S = np.zeros(n_parents, dtype=np.int)

            _S[0] = random.choice(feasible)

            nearest_neighbors = np.argsort(D[_S[0]])[:min(self.n_neighbors, len(infeasible))]
            _S[1] = random.choice(nearest_neighbors)

            S.append(_S)

        S = np.row_stack(S)

        return S


class ReferenceDirectionSurvivalPlus(Survival):
    def __init__(self, ref_dirs):
        super().__init__(False)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        infeasible = pop.get("CV")[:, 0] > 0

        # split the population into feasible and infeasible
        pop_infeasible = pop[infeasible]
        pop = pop[np.logical_not(infeasible)]

        # get the objective values of population
        F = pop.get("F")

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)
        pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche', dist_to_niche)

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(F[last_front, :], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

            # -----------------------------------
            # Now archive also infeasible solutions
            # -----------------------------------

            if len(pop_infeasible) > 0:

                F, CV = pop_infeasible.get("F", "CV")

                fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
                non_dominated, last_front = fronts[0], fronts[-1]

                # associate individuals to niches
                niche_of_individuals, _ = associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)
                pop_infeasible.set('rank', rank, 'niche', niche_of_individuals)


                # if there is only one front
                if len(fronts) == 1:
                    n_remaining = min(n_survive, len(last_front))
                    until_last_front = np.array([], dtype=np.int)
                    niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

                # if some individuals already survived
                else:
                    until_last_front = np.concatenate(fronts[:-1])
                    niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                    n_remaining = min(n_survive - len(until_last_front), len(last_front))

                print(len(fronts))
                
                S = infeasible_niching(F[last_front, :], CV[last_front, :], n_remaining, niche_count,
                                       niche_of_individuals[last_front])
                                       
                survivors = np.concatenate((until_last_front, last_front[S].tolist()))

                pop_infeasible = pop_infeasible[survivors]

        return pop.merge(pop_infeasible)


def infeasible_niching(F, CV, n_remaining, niche_count, niche_of_individuals):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(F.shape[0], True)

    while len(survivors) < n_remaining:

        # all niches where new individuals can be assigned to
        next_niches_list = np.unique(niche_of_individuals[mask])

        # pick a niche with minimum assigned individuals - break tie if necessary
        next_niche_count = niche_count[next_niches_list]
        next_niche = np.where(next_niche_count == next_niche_count.min())[0]
        next_niche = next_niches_list[next_niche]
        next_niche = next_niche[random.randint(0, len(next_niche))]

        # indices of individuals that are considered and assign to next_niche
        next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

        # shuffle to break random tie (equal perp. dist) or select randomly
        next_ind = random.shuffle(next_ind)

        if niche_count[next_niche] == 0:
            next_ind = next_ind[np.argmin(CV[next_ind])]
            #next_ind = next_ind[0]
        else:
            # already randomized through shuffling
            next_ind = next_ind[0]

        mask[next_ind] = False
        survivors.append(int(next_ind))

        niche_count[next_niche] += 1

    return survivors
