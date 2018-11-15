import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.rand import random
from pymoo.util.decomposition import DoNotKnow
from pymoo.util.display import disp_multi_objective


class MOEAD(GeneticAlgorithm):
    def __init__(self,
                 ref_dirs,
                 n_neighbors=15,
                 #decomposition=PenaltyBasedBoundaryInterception(theta=0.00000001),
                 decomposition=DoNotKnow(),
                 prob_neighbor_mating=0.7,
                 **kwargs):

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=0.9, eta_cross=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(eta_mut=15))
        set_if_none(kwargs, 'selection', None)
        set_if_none(kwargs, 'mutation', None)
        set_if_none(kwargs, 'survival', None)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

        # initialized when problem is known
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        self.ideal_point = None

    def _initialize(self):
        pop = super()._initialize()
        self.ideal_point = np.min(pop.F, axis=0)
        return pop

    def _next(self, pop):

        print(np.unique(pop.X, axis=0).shape[0])

        # iterate for each member of the population in random order
        for i in random.perm(self.pop_size):

            if random.random() < self.prob_neighbor_mating:
                parents = self.neighbors[i, :][random.perm(self.n_neighbors)][:self.crossover.n_parents]
            else:
                parents = np.arange(self.pop_size)[random.perm(self.pop_size)][:self.crossover.n_parents]

            # do recombination and create an offspring
            X = self.crossover.do(self.problem, pop.X[parents, None, :])
            X = self.mutation.do(self.problem, X)
            X = X[0, :]

            # evaluate the offspring
            F, _ = self.evaluator.eval(self.problem, X)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, F]), axis=0)

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            # calculate the decomposed values for each neighbor
            FV = self.decomposition.do(pop.F[N, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self.decomposition.do(F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop.F[N[I], :] = F
            pop.X[N[I], :] = X
