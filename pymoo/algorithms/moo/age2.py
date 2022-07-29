from pymoo.docs import parse_doc_string

try:
    import numba
    from numba import jit
except:
    raise Exception("Please install numba to use AGEMOEA2: pip install numba")

import numpy as np


from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.age import AGEMOEASurvival
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput


class AGEMOEA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """
        Adapted from:
        Panichella, A. (2022). An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization.
        Proceedings of the 2022 Genetic and Evolutionary Computation Conference (GECCO 2022).
        https://doi.org/10.1145/3512290.3528732

        @author: Annibale Panichella

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
                         survival=AGEMOEA2Survival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)
        self.tournament_type = 'comp_by_rank_and_crowding'


@jit(fastmath=True)
def project_on_manifold(point, p):
    dist = sum(point[point > 0] ** p) ** (1/p)
    return np.multiply(point, 1 / dist)


def find_zero(point, n, precision):
    x = 1

    past_value = x
    for i in range(0, 100):

        # Original function
        f = 0.0
        for obj_index in range(0, n):
            if point[obj_index] > 0:
                f += np.power(point[obj_index], x)

        f = np.log(f)

        # Derivative
        numerator = 0
        denominator = 0
        for obj_index in range(0, n):
            if point[obj_index] > 0:
                numerator = numerator + np.power(point[obj_index], x) * np.log(point[obj_index])
                denominator = denominator + np.power(point[obj_index], x)

        if denominator == 0:
            return 1

        ff = numerator / denominator

        # zero of function
        x = x - f / ff

        if abs(x - past_value) <= precision:
            break
        else:
            paste_value = x  # update current point

    if isinstance(x, complex):
        return 1
    else:
        return x


class AGEMOEA2Survival(AGEMOEASurvival):

    @staticmethod
    def compute_geometry(front, extreme, n):
        m, _ = np.shape(front)

        # approximate p(norm)
        d = np.zeros(m)
        for i in range(0, m):
            d[i] = sum(front[i] ** 2) ** 0.5

        d[extreme] = np.inf
        index = np.argmin(d)

        p = find_zero(front[index], n, 0.001)

        if np.isnan(p) or p <= 0.1:
            p = 1.0
        elif p > 20:
            p = 20.0  # avoid numpy underflow

        return p

    @staticmethod
    @jit(fastmath=True)
    def pairwise_distances(front, p):
        m, n = front.shape
        projected_front = front.copy()

        for index in range(0, m):
            projected_front[index] = project_on_manifold(front[index], p=p)

        distances = np.zeros((m, m), dtype=numba.float64)

        if 0.95 < p < 1.05:
            for row in range(0, m - 1):
                for column in range(row + 1, m):
                    distances[row][column] = sum(np.abs(projected_front[row] - projected_front[column]) ** 2) ** 0.5

        else:
            for row in range(0, m-1):
                for column in range(row+1, m):
                    mid_point = projected_front[row] * 0.5 + projected_front[column] * 0.5
                    mid_point = project_on_manifold(mid_point, p)

                    distances[row][column] = sum(np.abs(projected_front[row] - mid_point) ** 2) ** 0.5 + \
                                            sum(np.abs(projected_front[column] - mid_point) ** 2) ** 0.5

        return distances + distances.T

parse_doc_string(AGEMOEA2.__init__)
