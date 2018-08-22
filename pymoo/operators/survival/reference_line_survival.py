import numpy as np
from numpy.linalg import LinAlgError

from pymoo.model.survival import Survival
from pymoo.rand import random
from pymoo.util.non_dominated_rank import NonDominatedRank


class ReferenceLineSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__()
        self.ref_dirs = ref_dirs

        self.extreme_points = None
        self.intercepts = None
        self.ideal_point = None
        self.asf = None

    def _do(self, pop, n_survive, out=None, **kwargs):

        # calculate the fronts of the population
        fronts = NonDominatedRank.calc_as_fronts(pop.F)
        non_dom = fronts[0]

        # find or usually update the new ideal point
        if self.ideal_point is None:
            self.ideal_point = np.min(pop.F, axis=0)
        else:
            self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], pop.F], axis=0), axis=0)

        # calculate the worst point of feasible individuals
        worst_point = np.max(pop.F, axis=0)
        # calculate the nadir point from non dominated individuals
        nadir_point = np.max(pop.F[non_dom, :], axis=0)

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points(pop.F, self.ideal_point, extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        self.intercepts = get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)

        # find indices of all indices that have to be considered in the following
        rank = np.full(pop.size(), -1)
        I = []
        for k, front in enumerate(fronts):

            # set the rank for the individuals
            rank[front] = k

            # see if we need the front or not
            if len(I) + len(front) >= n_survive:
                break
            else:
                I.extend(front)

        # filter the population directly by all individuals to consider
        pop.filter(I + front)
        rank = rank[I + front]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche = associate_to_niches(pop.F, self.ref_dirs, self.ideal_point,
                                                                  self.intercepts)

        # if a splitting of the last front is not necessary
        if pop.size() == n_survive:
            survival = np.arange(pop.size())

        # otherwise we have to select using the niching
        else:

            # the first non-dominated solutions are surviving for sure
            survival = np.arange(len(I)).tolist()

            # the last front survivors need to be investigated
            last_front = np.arange(len(survival), pop.size())

            # if the last front needs to be splitted
            n_remaining = n_survive - len(survival)

            # for each reference direction the niche count
            niche_count = np.zeros(len(self.ref_dirs))
            for i in niche_of_individuals[survival]:
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
                survival.append(int(last_front[next_ind]))

                niche_count[next_niche] += 1
                n_remaining -= 1

        # set attributes globally for other modules
        if out is not None:
            out['rank'] = rank[survival]
            out['niche'] = niche_of_individuals[survival]
            out['dist_to_niche'] = dist_to_niche[survival]

        # now truncate the population
        pop.filter(survival)


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
        intercepts = np.linalg.solve(extreme_points - ideal_point, np.ones(extreme_points.shape[1]))

        intercepts = (1 / intercepts)

        """
        if np.any(intercepts < 1e-6):
            raise LinAlgError()
        else:
            intercepts = (1 / intercepts)
        """


    except LinAlgError:
        # set to zero which will be handled later
        intercepts = nadir_point

    # if even that point is too small
    if np.any(intercepts < 1e-6):
        intercepts = worst_point

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
