import numpy as np
from numpy.linalg import LinAlgError

from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.rand import random
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs

        self.extreme_points = None
        self.intercepts = None
        self.ideal_point = None

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

            # find or usually update the new ideal point - from feasible solutions
            if self.ideal_point is None:
                self.ideal_point = np.min(F, axis=0)
            else:
                self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], F], axis=0), axis=0)

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=1e-10).do(F, return_rank=True, n_stop_if_ranked=n_survive_feasible)
            non_dominated = fronts[0]

            # calculate the worst point of feasible individuals
            worst_point = np.max(F, axis=0)
            # calculate the nadir point from non dominated individuals
            nadir_point = np.max(F[non_dominated, :], axis=0)

            # consider only the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # find the extreme points for normalization
            self.extreme_points = get_extreme_points(F, self.ideal_point, extreme_points=self.extreme_points)

            # find the intercepts for normalization and do backup if gaussian elimination fails
            self.intercepts = get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)

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
            rank = np.concatenate([rank, 1e16 * np.ones(n_infeasible)])
            niche_of_individuals = np.concatenate([niche_of_individuals, -1 * np.ones(n_infeasible)])
            dist_to_niche = np.concatenate([dist_to_niche, 1e30 * np.ones(n_infeasible)])

        # set attributes globally for other modules
        if out is not None:
            out['rank'] = rank
            out['niche'] = niche_of_individuals
            out['dist_to_niche'] = dist_to_niche

        # now truncate the population
        pop.filter(survivors)


def get_extreme_points(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e-6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([_F, extreme_points], axis=0)

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max((_F - ideal_point) / asf[:, None, :], axis=2)
    extreme_points = _F[np.argmin(F_asf, axis=1), :]

    return extreme_points


def get_intercepts(extreme_points, ideal_point, nadir_point, worst_point):
    # normalization of the points in the new space
    nadir_point -= ideal_point
    worst_point -= ideal_point

    try:
        # find the intercepts using gaussian elimination
        intercepts = 1 / np.linalg.solve(extreme_points - ideal_point, np.ones(extreme_points.shape[1]))

    except LinAlgError:
        # set to zero which will be handled later
        intercepts = nadir_point

    # if even that point is too small
    if np.any(intercepts < 1e-6):
        intercepts = worst_point

    # if also the worst point is very small we set it to a small value, to avoid division by zero
    intercepts[intercepts < 1e-16] = 1e-16

    return intercepts


def get_intercepts_mod(extreme_points, ideal_point, nadir_point, worst_point):
    # normalization of the points in the new space
    nadir_point -= ideal_point
    worst_point -= ideal_point

    gaussian_elimination_failed = False

    try:
        # find the intercepts using gaussian elimination
        intercepts = 1 / np.linalg.solve(extreme_points - ideal_point, np.ones(extreme_points.shape[1]))
    except LinAlgError:
        gaussian_elimination_failed = True

    # in case the gaussian elimination failed (this will happen when ever the matrix is singular)
    if gaussian_elimination_failed:
        intercepts = nadir_point

    else:

        # gaussian elimination is degenerated - (negative intercepts, too small, too large)
        b = np.logical_or(intercepts < 1e-6, intercepts > 1e6)

        # replace these values with nadir
        if np.any(b):
            intercepts[b] = nadir_point[b]

    # if even that point is too small still (after taking the nadir point usually - if only one dominated point it is 0)
    b = intercepts < 1e-6
    intercepts[b] = worst_point[b]

    return intercepts


def associate_to_niches(F, niches, ideal_point, intercepts, utopianEpsilon=-0.001):

    # normalize by ideal point and intercepts
    N = (F - ideal_point) / intercepts

    # make sure that no values are 0. (subtracting a negative epsilon)
    N -= utopianEpsilon

    dist_matrix = calc_perpendicular_dist_matrix(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = np.min(dist_matrix, axis=1)

    return niche_of_individuals, dist_to_niche


def calc_perpendicular_dist_matrix(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(N), len(ref_dirs)))

    return matrix
