import numpy as np
from scipy.spatial.distance import cdist

from pymoo.cython.decomposition import PenaltyBasedBoundaryIntersection, Tchebicheff
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective


class MOEAD(GeneticAlgorithm):
    def __init__(self,
                 ref_dirs,
                 n_neighbors=15,
                 decomposition='auto',
                 prob_neighbor_mating=0.7,
                 **kwargs):

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=0.9, eta_cross=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(eta_mut=15))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

        # initialized when problem is known
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

    def _initialize(self):

        if isinstance(self.decomposition, str):

            # for one or two objectives use tchebi otherwise pbi
            if self.decomposition == 'auto':
                if self.problem.n_obj <= 2:
                    str_decomposition = 'tchebi'
                else:
                    str_decomposition = 'pbi'
            else:
                str_decomposition = self.decomposition

            if str_decomposition == 'tchebi':
                self._decomposition = Tchebicheff()
            elif str_decomposition == 'pbi':
                self._decomposition = PenaltyBasedBoundaryIntersection(5)

        else:
            self._decomposition = self.decomposition

        pop = super()._initialize()
        self.ideal_point = np.min(pop.get("F"), axis=0)

        return pop

    def _next(self, pop):

        # iterate for each member of the population in random order
        for i in random.perm(len(pop)):

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            if random.random() < self.prob_neighbor_mating:
                parents = N[random.perm(self.n_neighbors)][:self.crossover.n_parents]
            else:
                parents = random.perm(self.pop_size)[:self.crossover.n_parents]

            # do recombination and create an offspring
            off = self.crossover.do(self.problem, pop, parents[None, :])
            off = self.mutation.do(self.problem, off)
            off = off[random.randint(0, len(off))]

            # evaluate the offspring
            self.evaluator.eval(self.problem, off)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self._decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off

        return pop
