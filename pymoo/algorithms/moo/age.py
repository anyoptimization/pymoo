try:
    import numba
    from numba import jit
except:
    raise Exception("Please install numba to use AGEMOEA: pip install numba")

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.misc import has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================

class AGEMOEA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """
        Adapted from:
        Panichella, A. (2019). An adaptive evolutionary algorithm based on non-euclidean geometry for many-objective
        optimization. GECCO 2019 - Proceedings of the 2019 Genetic and Evolutionary Computation Conference, July, 595–603.
        https://doi.org/10.1145/3321707.3321839

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
                         survival=AGEMOEASurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)
        self.default_termination = DefaultMultiObjectiveTermination()

        self.tournament_type = 'comp_by_rank_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

class AGEMOEASurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective values
        F = pop.get("F")

        N = n_survive

        # Non-dominated sorting
        fronts = self.nds.do(F, n_stop_if_ranked=N)

        # get max int value
        max_val = np.iinfo(int).max

        # initialize population ranks with max int value
        front_no = np.full(F.shape[0], max_val, dtype=int)

        # assign the rank to each individual
        for i, fr in enumerate(fronts):
            front_no[fr] = i

        pop.set("rank", front_no)

        # get the index of the front to be sorted and cut
        max_f_no = np.max(front_no[front_no != max_val])

        # keep fronts that have lower rank than the front to cut
        selected: np.ndarray = front_no < max_f_no

        n_ind, _ = F.shape

        # crowding distance is positive and has to be maximized
        crowd_dist = np.zeros(n_ind)

        # get the first front for normalization
        front1 = F[front_no == 0, :]

        # follows from the definition of the ideal point but with current non dominated solutions
        ideal_point = np.min(front1, axis=0)

        # Calculate the crowding distance of the first front as well as p and the normalization constants
        crowd_dist[front_no == 0], p, normalization = self.survival_score(front1, ideal_point)
        for i in range(1, max_f_no):  # skip first front since it is normalized by survival_score
            front = F[front_no == i, :]
            m, _ = front.shape
            front = front / normalization
            crowd_dist[front_no == i] = 1. / self.minkowski_distances(front, ideal_point[None, :], p=p).squeeze()

        # Select the solutions in the last front based on their crowding distances
        last = np.arange(selected.shape[0])[front_no == max_f_no]
        rank = np.argsort(crowd_dist[last])[::-1]
        selected[last[rank[: N - np.sum(selected)]]] = True

        pop.set("crowding", crowd_dist)

        # return selected solutions, number of selected should be equal to population size
        return pop[selected]

    def survival_score(self, front, ideal_point):
        front = np.round(front, 12, out=front)
        m, n = front.shape
        crowd_dist = np.zeros(m)

        if m < n:
            p = 1
            normalization = np.max(front, axis=0)
            return crowd_dist, p, normalization

        # shift the ideal point to the origin
        front = front - ideal_point

        # Detect the extreme points and normalize the front
        extreme = find_corner_solutions(front)
        front, normalization = normalize(front, extreme)

        # set the distance for the extreme solutions
        crowd_dist[extreme] = np.inf
        selected = np.full(m, False)
        selected[extreme] = True

        p = self.compute_geometry(front, extreme, n)

        nn = np.linalg.norm(front, p, axis=1)
        distances = self.pairwise_distances(front, p) / nn[:, None]

        neighbors = 2
        remaining = np.arange(m)
        remaining = list(remaining[~selected])
        for i in range(m - np.sum(selected)):
            mg = np.meshgrid(np.arange(selected.shape[0])[selected], remaining, copy=False, sparse=False)
            D_mg = distances[tuple(mg)]  # avoid Numpy's future deprecation of array special indexing

            if D_mg.shape[1] > 1:
                # equivalent to mink(distances(remaining, selected),neighbors,2); in Matlab
                maxim = np.argpartition(D_mg, neighbors - 1, axis=1)[:, :neighbors]
                tmp = np.sum(np.take_along_axis(D_mg, maxim, axis=1), axis=1)
                index: int = np.argmax(tmp)
                d = tmp[index]
            else:
                index = D_mg[:, 0].argmax()
                d = D_mg[index, 0]

            best = remaining.pop(index)
            selected[best] = True
            crowd_dist[best] = d

        return crowd_dist, p, normalization

    @staticmethod
    def compute_geometry(front, extreme, n):
        # approximate p(norm)
        d = point_2_line_distance(front, np.zeros(n), np.ones(n))
        d[extreme] = np.inf
        index = np.argmin(d)

        p = np.log(n) / np.log(1.0 / np.mean(front[index, :]))

        if np.isnan(p) or p <= 0.1:
            p = 1.0
        elif p > 20:
            p = 20.0  # avoid numpy underflow

        return p

    @staticmethod
    @jit(fastmath=True)
    def pairwise_distances(front, p):
        m = np.shape(front)[0]
        distances = np.zeros((m, m))
        for i in range(m):
            distances[i] = np.sum(np.abs(front[i] - front) ** p, 1) ** (1 / p)

        return distances

    @staticmethod
    @jit(fastmath=True)
    def minkowski_distances(A, B, p):
        m1 = np.shape(A)[0]
        m2 = np.shape(B)[0]
        distances = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                distances[i][j] = sum(np.abs(A[i] - B[j]) ** p) ** (1 / p)

        return distances


@jit(nopython=True, fastmath=True)
def find_corner_solutions(front):
    """Return the indexes of the extreme points"""

    m, n = front.shape

    if m <= n:
        return np.arange(m)

    # let's define the axes of the n-dimensional spaces
    W = 1e-6 + np.eye(n)
    r = W.shape[0]
    indexes = np.zeros(n, dtype=numba.intp)
    selected = np.zeros(m, dtype=numba.boolean)
    for i in range(r):
        dists = point_2_line_distance(front, np.zeros(n), W[i, :])
        dists[selected] = np.inf  # prevent already selected to be reselected
        index = np.argmin(dists)
        indexes[i] = index
        selected[index] = True
    return indexes


@jit(fastmath=True)
def point_2_line_distance(P, A, B):
    d = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        pa = P[i] - A
        ba = B - A
        t = np.dot(pa, ba) / np.dot(ba, ba)
        d[i] = sum((pa - t * ba) ** 2)

    return d


# =========================================================================================================
# Normalization
# =========================================================================================================

def normalize(front, extreme):
    m, n = front.shape

    if len(extreme) != len(np.unique(extreme, axis=0)):
        normalization = np.max(front, axis=0)
        front = front / normalization
        return front, normalization

    # Calculate the intercepts of the hyperplane constructed by the extreme
    # points and the axes

    try:
        hyperplane = np.linalg.solve(front[extreme], np.ones(n))
        if any(np.isnan(hyperplane)) or any(np.isinf(hyperplane)) or any(hyperplane < 0):
            normalization = np.max(front, axis=0)
        else:
            normalization = 1. / hyperplane
            if any(np.isnan(normalization)) or any(np.isinf(normalization)):
                normalization = np.max(front, axis=0)
    except np.linalg.LinAlgError:
        normalization = np.max(front, axis=0)

    normalization[normalization == 0.0] = 1.0

    # Normalization
    front = front / normalization

    return front, normalization

parse_doc_string(AGEMOEA.__init__)
