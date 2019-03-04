import numpy as np

from pymoo.algorithms.genetic_algorithm import default_is_duplicate
from pymoo.algorithms.nsga3 import get_nadir_point
from pymoo.experimental.nsgadss import calc_crowding_distance
from pymoo.model.survival import Survival
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.misc import cdist
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem, ConvexProblem, DTLZ2


class DSSSurvival(Survival):
    def __init__(self):
        super().__init__(True)
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

        self.n_archive = 2000
        self.archive = None

    def _do(self, pop, n_survive, algorithm=None, **kwargs):

        if True:

            if self.archive is None:
                self.archive = pop
            else:
                self.archive = self.archive.merge(pop)

            # get the function values of the current archive for operations
            F = self.archive.get("F")

            # filter out all the duplicate solutions
            I = np.logical_not(default_is_duplicate(F))
            self.archive, F = self.archive[I], F[I]

            # get only the non-dominated solutions
            I = NonDominatedSorting().do(F, only_non_dominated_front=True)
            self.archive, F = self.archive[I], F[I]

            if len(self.archive) > self.n_archive:
                cd = calc_crowding_distance(F)
                self.archive = self.archive[np.argsort(cd)[::-1][:self.n_archive]]

            # attributes to be set after the survival
            pop.merge(self.archive)

        F = pop.get("F")

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting(epsilon=1e-10).do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = non_dominated[get_extreme_points(F[non_dominated, :], self.ideal_point)]

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(F[self.extreme_points], self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        # associate individuals to niches
        pop.set('rank', rank)

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            for i in range(len(fronts)):
                fronts[i] = np.array([j for j in fronts[i] if j not in self.extreme_points])

            survivors = np.unique(self.extreme_points).tolist()

            for k, front in enumerate(fronts):

                # current front sorted by crowding distance if splitting
                if len(survivors) + len(front) > n_survive:
                    I = selection(survivors, list(front), F, (n_survive - len(survivors)))
                    survivors.extend(I)

                # otherwise take the whole front unsorted
                else:
                    # extend the survivors by all or selected individuals
                    survivors.extend(front)

            pop = pop[survivors]

        return pop


def selection(surviving, not_surviving, F, n_remaining):
    val = []
    D = cdist(F, F)

    for i in range(n_remaining):
        I = not_surviving[np.argmax(np.min(D[not_surviving, :][:, surviving], 1))]
        surviving.append(I)
        not_surviving.remove(I)

        val.append(I)

    return val


def get_extreme_points(F, ideal_point):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    return I


def normalize(F, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon
    N = (F - utopian_point) / (nadir_point - utopian_point)
    return N


problem = get_problem("carside")

#problem = ConvexProblem(DTLZ2(n_var=12, n_obj=3))

n_gen = 400
pop_size = 91
ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=12, scaling=1.0).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(UniformReferenceDirectionFactory(3, n_partitions=70, scaling=1.0).do())

res = minimize(problem,
               method='nsga3',
               method_args={
                   'pop_size': 91,
                   'ref_dirs': ref_dirs,
                  'survival': DSSSurvival()
               },
               termination=('n_gen', n_gen),
              # pf=pf,
               save_history=True,
               seed=31,
               disp=True)

#plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
plotting.plot(res.F, labels=["F"])
