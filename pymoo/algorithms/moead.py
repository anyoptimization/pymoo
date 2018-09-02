import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymoo.util.decomposition import decompose
from pymoo.util.display import disp_multi_objective


class MOEAD(GeneticAlgorithm):
    def __init__(self,
                 ref_dirs,
                 n_neighbors=15,
                 decomposition="cython_pbi",
                 prob_neighbor_mating=0.7,
                 **kwargs):

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=0.9, eta_cross=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(eta_mut=15))
        set_if_none(kwargs, 'selection', None)
        set_if_none(kwargs, 'mutation', None)
        set_if_none(kwargs, 'survival', None)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

        # initialized when problem is known
        self.ref_dirs = ref_dirs

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        self.ideal_point = None

    def _initialize(self):
        pop = super()._initialize()
        self.ideal_point = np.min(pop.F, axis=0)
        return pop

    def _next(self, pop):

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

            batch = False

            if batch:

                # all neighbors of this individual and corresponding weights
                N = self.neighbors[i, :]
                w = self.ref_dirs[N, :]

                # calculate the decomposed values for each neighbor
                FV = decompose(pop.F[N, :], w, method=self.decomposition, ideal_point=self.ideal_point, theta=0.01)
                off_FV = decompose(F[None, :], w, method=self.decomposition, ideal_point=self.ideal_point, theta=0.01)

                # get the absolute index in F where offspring is better than the current F (decomposed space)
                neighbors_to_update = N[off_FV < FV]
                pop.F[neighbors_to_update, :] = F
                pop.X[neighbors_to_update, :] = X


            else:

                for neighbor in self.neighbors[i][random.perm(self.n_neighbors)]:

                    # the weight vector of this neighbor
                    w = self.ref_dirs[neighbor, :]

                    FV = decompose(pop.F[[neighbor], :], w, method=self.decomposition, ideal_point=self.ideal_point,
                                   theta=0.01)
                    off_FV = decompose(F[None, :], w, method=self.decomposition, ideal_point=self.ideal_point,
                                       theta=0.01)

                    # replace if better
                    if np.all(off_FV < FV):
                        pop.X[neighbor, :], pop.F[neighbor, :] = X, F
