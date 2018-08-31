import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.algorithms.unsga3 import comp_by_rank_and_ref_line_dist
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.reference_line_survival import get_extreme_points, get_intercepts, associate_to_niches
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymop.util import get_uniform_weights


class AdaptiveNSGA3(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 ref_dirs=None,
                 prob_cross=1.0,
                 eta_cross=20,
                 prob_mut=None,
                 eta_mut=30,
                 **kwargs):

        self.ref_dirs = ref_dirs

        set_if_none(kwargs, 'pop_size', pop_size)
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=prob_cross, eta_cross=eta_cross))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=prob_mut, eta_mut=eta_mut))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

    def _initialize(self):
        pop = super()._initialize()

        # if survival not define differently
        if self.survival is None:

            # if ref dirs are not initialized do it based on the population size
            if self.ref_dirs is None:
                self.ref_dirs = get_uniform_weights(self.pop_size, self.problem.n_obj)

            # set the survival method itself
            self.survival = AdaptiveReferenceLineSurvival(self.ref_dirs)

        # call the survival to initialize ideal point and so on - does not do a actual survival
        self.survival.do(pop, self.pop_size, out=self.D, **self.D)

        return pop


class AdaptiveReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs

        self.extreme_points = None
        self.intercepts = None
        self.ideal_point = None
        self.nadir_point = None

        self.adam_alpha = 0.1
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.9
        self.adam_epsilon = 1e-8
        self.alpha = 0.1

        self.adam_m = None
        self.adam_v = None
        self.adam_theta = None
        self.adam_t = None

    def _do(self, pop, n_survive, out=None, **kwargs):

        # convert to integer for later usage
        n_survive = int(n_survive)

        # first split by feasibility for normalization
        feasible, infeasible = split_by_feasibility(pop)

        # number of survivors from the feasible population
        # in case of having not enough feasible solution all feasible will survive
        if len(feasible) < n_survive:
            n_survive_feasible = len(feasible)
        else:
            n_survive_feasible = n_survive

        # attributes to be set after the survival
        survivors, rank, niche_of_individuals, dist_to_niche = [], [], [], []

        # if there are feasible solutions to survive
        if len(feasible) > 0:

            # consider only feasible solutions form now on
            F = pop.F[feasible, :]

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=Mathematics.EPS).do(F, return_rank=True,
                                                                            n_stop_if_ranked=n_survive_feasible)

            # find or usually update the new ideal point - from feasible solutions
            if self.ideal_point is None:
                self.ideal_point = np.min(F, axis=0)
            else:
                self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], F], axis=0), axis=0)




            self.extreme_points = get_extreme_points(F, self.ideal_point, extreme_points=self.extreme_points)


            adam = True

            if adam:
                # now update the nadir point to the correct direction - take the worst of initial population
                if self.nadir_point is None:
                    self.nadir_point = np.max(F, axis=0)
                    self.adam_m = np.zeros(F.shape[1])
                    self.adam_v = np.zeros(F.shape[1])
                    self.adam_t = 0

                else:

                    self.adam_t += 1

                    # estimate the nadir point using the payoff table estimation
                    estimated_nadir_point = np.max(self.extreme_points, axis=0)

                    print("estimated: ", estimated_nadir_point)

                    # now update the nadir point using the adam update rule - this is our "gradient"
                    g = (estimated_nadir_point - self.nadir_point)
                    # print(g)

                    self.adam_m = self.adam_beta_1 * self.adam_m + (1 - self.adam_beta_1) * g
                    self.adam_v = self.adam_beta_2 * self.adam_v + (1 - self.adam_beta_2) * np.square(g)

                    _m = self.adam_m / (1 - np.power(self.adam_beta_1, self.adam_t))
                    _v = self.adam_v / (1 - np.power(self.adam_beta_2, self.adam_t))

                    self.nadir_point = self.nadir_point + self.alpha * _m

            else:

                if self.nadir_point is None:
                    self.nadir_point = np.max(F, axis=0)
                else:
                    estimated_nadir_point = np.max(self.extreme_points, axis=0)
                    self.nadir_point = self.nadir_point + (estimated_nadir_point - self.nadir_point) / 10
                    #self.nadir_point = estimated_nadir_point
                    print("estimated: ", estimated_nadir_point)


            self.intercepts = self.nadir_point - self.ideal_point

            if out is not None:
                out['nadir_point'] = self.nadir_point

            print("used: ", self.intercepts + self.ideal_point)
            print("=" * 20)

            # consider only the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point,
                                                                      self.intercepts)

            # if a splitting of the last front is not necessary
            if F.shape[0] == n_survive_feasible:
                _survivors = np.arange(F.shape[0])

            # otherwise we have to select using niching
            else:

                # number of individuals taken by fronts - if only one front niching over all solutions
                if len(fronts) == 1:
                    n_until_splitting_front = 0
                else:
                    n_until_splitting_front = len(np.concatenate(fronts[:-1]))
                _survivors = np.arange(n_until_splitting_front).tolist()

                # last front to be assigned to
                last_front = np.arange(n_until_splitting_front, F.shape[0])

                # if the last front needs to be splitted
                n_remaining = n_survive_feasible - len(_survivors)

                # for each reference direction the niche count
                niche_count = np.zeros(len(self.ref_dirs))
                for i in niche_of_individuals[_survivors]:
                    niche_count[i] += 1

                # relative index to dist and the niches just of the last front
                lf_dist_to_niche = dist_to_niche[last_front]
                lf_niche_of_individuals = niche_of_individuals[last_front]

                # boolean array of elements that are considered for each iteration
                remaining_last_front = np.full(len(last_front), True)

                while n_remaining > 0:

                    # all niches where new individuals can be assigned to
                    next_niches_list = np.unique(lf_niche_of_individuals[remaining_last_front])

                    # pick a niche with minimum assigned individuals - break tie if necessary
                    next_niche_count = niche_count[next_niches_list]
                    next_niche = np.where(next_niche_count == next_niche_count.min())[0]
                    next_niche = next_niches_list[next_niche]
                    next_niche = next_niche[random.randint(0, len(next_niche))]

                    # indices of individuals that are considered and assign to next_niche
                    next_ind = np.where(np.logical_and(lf_niche_of_individuals == next_niche, remaining_last_front))[0]

                    if len(next_ind) == 1:
                        next_ind = next_ind[0]
                    elif niche_count[next_niche] == 0:
                        next_ind = next_ind[np.argmin(lf_dist_to_niche[next_ind])]
                    else:
                        # not sorted so randomly the first is fine here
                        next_ind = next_ind[0]
                        # next_ind = next_ind[random.randint(0, len(next_ind))]

                    remaining_last_front[next_ind] = False
                    _survivors.append(int(last_front[next_ind]))

                    niche_count[next_niche] += 1
                    n_remaining -= 1

            # reindex the survivors to the absolute index
            survivors = feasible[I[_survivors]]

            # save the attributes for surviving individuals
            rank = _rank[I[_survivors]]
            niche_of_individuals = niche_of_individuals[_survivors]
            dist_to_niche = dist_to_niche[_survivors]

        # if we need to fill up with infeasible solutions - we do so. Also, the data structured need to be reindexed
        n_infeasible = n_survive - len(survivors)
        if n_infeasible > 0:
            survivors = np.concatenate([survivors, infeasible[:n_infeasible]])
            rank = np.concatenate([rank, Mathematics.INF * np.ones(n_infeasible)])
            niche_of_individuals = np.concatenate([niche_of_individuals, -1 * np.ones(n_infeasible)])
            dist_to_niche = np.concatenate([dist_to_niche, Mathematics.INF * np.ones(n_infeasible)])

        # set attributes globally for other modules
        if out is not None:
            out['rank'] = rank
            out['niche'] = niche_of_individuals
            out['dist_to_niche'] = dist_to_niche

        # now truncate the population
        pop.filter(survivors)
