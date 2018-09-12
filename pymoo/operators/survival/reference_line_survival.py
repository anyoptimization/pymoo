import numpy as np
from numpy.linalg import LinAlgError

from pymoo.cython.my_math import cython_calc_perpendicular_distance
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.rand import random
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
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

            # calculate the worst point of feasible individuals
            worst_point = np.max(F, axis=0)
            # calculate the nadir point from non dominated individuals
            nadir_point = np.max(F[non_dominated, :], axis=0)

            # find the extreme points for normalization
            self.extreme_points = get_extreme_points(F, self.ideal_point, extreme_points=self.extreme_points)

            # find the intercepts for normalization and do backup if gaussian elimination fails
            self.intercepts = get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)

            # index of the first n fronts form now on - including splitting front
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


def niching_vectorized(F, n_survive, niche_count, niche_of_individuals, dist_to_niche):
    survivors = []
    n_niches = len(niche_count)

    # maximum niche count of the last front
    max_niche_count = np.max(np.unique(niche_of_individuals, return_counts=True)[1])

    # stores for each niche the corresponding individuals - (-1=does not exist, -2=added before, i=otherwise)
    individuals_of_niches = np.full((n_niches, max_niche_count + 1), -1, dtype=np.int)

    # the minimum distance to the niche - just used to find the closest niche
    min_dist_closest_to_niche = np.full(n_niches, np.inf)

    # closest individual to niche - (NOTE: it is just an index to individuals_of_niches!)
    ind_closest_to_niche = np.full(n_niches, -1, dtype=np.int)

    # fill individuals_of_niches and ind_closest_to_niche with values
    counter = np.zeros(n_niches, dtype=np.int)
    for i in range(F.shape[0]):
        niche = niche_of_individuals[i]
        individuals_of_niches[niche, counter[niche]] = i

        if dist_to_niche[i] < min_dist_closest_to_niche[niche]:
            min_dist_closest_to_niche[niche] = dist_to_niche[i]
            ind_closest_to_niche[niche] = counter[niche]

        counter[niche] += 1

    # this mask is used to selected for the corresponding niches the next individuals
    mask = np.zeros(n_niches, dtype=np.int)

    # while we do have individuals to assign
    while len(survivors) < n_survive:

        # only consider niches that do have at least one individual assigned
        niches_remaining = np.where(individuals_of_niches[np.arange(n_niches), mask] != -1)[0]
        niche_count_remaining = niche_count[niches_remaining]

        # the minimum niche count of the remaining niches
        min_niche_count = np.min(niche_count_remaining)

        # get all niches with the minimum assignment
        selected_niches = niches_remaining[niche_count_remaining == min_niche_count]

        # remaining individuals which is the same as niches to select
        n_remaining = n_survive - len(survivors)

        # if only a few niches need to be select to fill up until n_survive - select the randomly
        if n_remaining < len(selected_niches):
            P = random.perm(len(selected_niches))
            selected_niches = selected_niches[P[:n_remaining]]

        # if we did not assign any individual
        if min_niche_count == 0:

            # select the closest individual to niche
            closest_individuals_of_niche = ind_closest_to_niche[selected_niches]
            S = individuals_of_niches[selected_niches, closest_individuals_of_niche]

            # since increasing does not work - here the selected are marked and skipped later
            individuals_of_niches[selected_niches, closest_individuals_of_niche] = -2

        else:
            # otherwise we just select one randomly through mask (values or not sorted)
            S = individuals_of_niches[selected_niches, mask[selected_niches]]
            # increase the mask by one for selected
            mask[selected_niches] += 1

        # if the mask points to an individual select by closest distance just point to the next
        mask[individuals_of_niches[np.arange(n_niches), mask] == -2] += 1

        # increase the niche count of the corresponding niche
        niche_count[selected_niches] += 1

        # save the individuals that survived
        survivors.extend(S)

    return survivors


def niching(F, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
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

        if len(next_ind) == 1:
            next_ind = next_ind[0]
        elif niche_count[next_niche] == 0:
            next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
        else:
            # not sorted so randomly the first is fine here
            next_ind = next_ind[0]
            # next_ind = next_ind[random.randint(0, len(next_ind))]

        mask[next_ind] = False
        survivors.append(int(next_ind))

        niche_count[next_niche] += 1

    return survivors


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

    use_nadir = False

    try:
        # find the intercepts using gaussian elimination
        plane = np.linalg.solve(extreme_points - ideal_point, np.ones(extreme_points.shape[1]))

        # if the plane
        if np.any(plane == 0):
            use_nadir = True
        else:
            intercepts = 1 / plane

    except LinAlgError:
        use_nadir = True

    if use_nadir:
        intercepts = nadir_point

    # if even that point is too small
    if np.any(intercepts < 1e-6):
        intercepts = worst_point

    # if also the worst point is very small we set it to a small value, to avoid division by zero
    intercepts[intercepts < 1e-16] = 1e-16

    return intercepts


def associate_to_niches(F, niches, ideal_point, intercepts, utopianEpsilon=-0.001):
    # normalize by ideal point and intercepts
    N = (F - ideal_point) / intercepts

    # make sure that no values are 0. (subtracting a negative epsilon)
    N -= utopianEpsilon

    # dist_matrix = calc_perpendicular_dist_matrix(N, niches)
    dist_matrix = cython_calc_perpendicular_distance(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

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


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count
