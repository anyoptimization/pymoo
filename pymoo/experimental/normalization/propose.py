import numpy as np

from pymoo.experimental.normalization.methods import naive_estimate_nadir_point
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.operators.survival.c_reference_line_survival import c_associate_to_niches
from pymoo.operators.survival.reference_line_survival import niching, calc_niche_count
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ProposeReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

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

            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], F], axis=0), axis=0)

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=Mathematics.EPS).do(F, return_rank=True,
                                                                            n_stop_if_ranked=n_survive_feasible)
            non_dominated, last_front = fronts[0], fronts[-1]

            # first estimate using this method
            if self.nadir_point is None:
                self.nadir_point = naive_estimate_nadir_point(self.ideal_point, F, fronts)

            try:
                # find the extreme points for normalization
                self.extreme_points = get_extreme_points_propose(F, self.ideal_point,
                                                                 extreme_points=self.extreme_points)
                self.nadir_point = hyperplane_nadir_point(self.extreme_points, self.ideal_point)

            except:
                pass


            # index of the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            niche_of_individuals, dist_to_niche = c_associate_to_niches(F, self.ref_dirs, self.ideal_point,
                                                                              self.nadir_point)

            # if a splitting of the last front is not necessary
            if F.shape[0] == n_survive_feasible:
                _survivors = np.arange(F.shape[0])

            # otherwise we have to select using niching
            else:

                _last_front = np.arange(len(I) - len(last_front), len(I))
                _until_last_front = np.arange(0, len(I) - len(last_front))

                _survivors = []
                n_remaining = n_survive_feasible
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

                if len(fronts) > 1:
                    _survivors.extend(_until_last_front)
                    niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[_until_last_front])
                    n_remaining -= len(_until_last_front)

                S = niching(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                            dist_to_niche[_last_front])

                # S = niching_vectorized(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                #                       dist_to_niche[_last_front])

                _survivors.extend(_last_front[S].tolist())

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
        if D is not None:
            D['rank'] = rank
            D['niche'] = niche_of_individuals
            D['dist_to_niche'] = dist_to_niche

        # now truncate the population
        pop.filter(survivors)


def get_extreme_points_propose(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e-6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([_F, extreme_points], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F / asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def hyperplane_nadir_point(extreme_points, ideal_point):
    # find the intercepts using gaussian elimination
    plane = np.linalg.solve(extreme_points - ideal_point, np.ones(extreme_points.shape[1]))

    # negative intercepts
    if np.any(plane <= 1e-6):
        raise Exception("Gaussian Elimination failed.")

    nadir_point = ideal_point + 1 / plane

    if np.any(nadir_point - ideal_point <= 1e-6):
        raise Exception("Gaussian Elimination estimation not significantly different from ideal point!")

    return nadir_point

