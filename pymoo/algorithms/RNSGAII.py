import scipy
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.survival import Survival
from pymoo.operators.crossover.bin_uniform_crossover import BinaryUniformCrossover
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.bin_bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.bin_random_sampling import BinaryRandomSampling
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival
from pymoo.rand import random
from pymoo.util.dominator import Dominator
from pymoo.util.misc import normalize, denormalize
from pymoo.util.non_dominated_rank import NonDominatedRank


import numpy as np

class RNSGAII(GeneticAlgorithm):
    def __init__(self, var_type, ref_points, epsilon=0.01, pop_size=100, verbose=1):

        if var_type == "real":
            super().__init__(
                pop_size,
                RealRandomSampling(),
                TournamentSelection(f_comp=comp_by_rank_and_crowding),
                SimulatedBinaryCrossover(),
                PolynomialMutation(),
                RSurvival(ref_points,epsilon),
                verbose=verbose
            )
        elif var_type == "binary":
            super().__init__(
                pop_size,
                BinaryRandomSampling(),
                TournamentSelection(f_comp=comp_by_rank_and_crowding),
                BinaryUniformCrossover(),
                BinaryBitflipMutation(),
                RSurvival(ref_points,epsilon),
                verbose=verbose,
                eliminate_duplicates=True
            )




class RSurvival(Survival):


    def __init__(self, ref_points, epsilon) -> None:
        super().__init__()
        self.ref_points = ref_points
        self.epsilon = epsilon


    def _do(self, pop, size, data, return_sorted_idx=False):

        fronts = NonDominatedRank.calc_as_fronts(pop.F, pop.G)
        rank = NonDominatedRank.calc_from_fronts(fronts)
        crowding = np.zeros(pop.F.shape[0])

        for front in fronts:


            n = len(front)
            m = len(self.ref_points)
            _, F_min, F_max = normalize(pop.F[front], return_bounds=True)

            # calculate the normalized distance matrix
            dist_matrix = np.zeros((n,m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = np.mean(np.square((pop.F[i,:] - self.ref_points[j,:]) / (F_max - F_min)))

            # assign the crowding rank
            crowding[front] = dist_matrix.argsort(axis=0).argsort(axis=0).min(axis=1)

        sorted_idx = sorted(range(pop.size()), key=lambda x: (rank[x], crowding[x]))

        if return_sorted_idx:
            return sorted_idx

        # now truncate the population
        sorted_idx = sorted_idx[:size]
        pop.filter(sorted_idx)
        rank = rank[sorted_idx]
        crowding = crowding[sorted_idx]

        if data is not None:
            data.rank = rank
            data.crowding = crowding

        return pop




def comp_by_rank_and_crowding(pop, indices, data):
    if len(indices) != 2:
        raise ValueError("Only implemented for binary tournament!")

    first = indices[0]
    second = indices[1]

    if data.rank[first] < data.rank[second]:
        return first
    elif data.rank[second] < data.rank[first]:
        return second
    else:
        if data.crowding[first] < data.crowding[second]:
            return first
        elif data.crowding[second] < data.crowding[first]:
            return second
        else:
            return indices[random.randint(0, 2)]


