import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymoo.util.decomposition import tchebi
from pymoo.util.display import disp_multi_objective
from pymop.util import get_uniform_weights


class MOEAD(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 n_neighbors=15,
                 prob=0.5,
                 weight=0.8,
                 method="binomial",
                 **kwargs):

        self.n_neighbors = n_neighbors

        set_if_none(kwargs, 'pop_size', pop_size)
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'crossover', DifferentialEvolutionCrossover(prob=prob, weight=weight, method=method))
        set_if_none(kwargs, 'selection', None)
        set_if_none(kwargs, 'mutation', None)
        set_if_none(kwargs, 'survival', None)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

        # initialized when problem is known
        self.weights = None
        self.neighbours = None
        self.ideal_point = None

    def _initialize(self):

        # weights to be used for decomposition
        self.weights = get_uniform_weights(self.pop_size, self.problem.n_obj)

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbours = np.argsort(cdist(self.weights, self.weights), axis=1, kind='quicksort')[:, :self.n_neighbors + 1]

        # set the initial ideal point
        pop = super()._initialize()
        self.ideal_point = np.min(pop.F, axis=0)

        return pop

    def _next(self, pop):

        # iterate for each member of the population
        for i in range(self.pop_size):

            # all neighbors shuffled (excluding the individual itself)
            neighbors = self.neighbours[i][1:][random.perm(self.n_neighbors - 1)]
            parents = np.concatenate([[i], neighbors[:self.crossover.n_parents - 1]])

            # do recombination and create an offspring
            X = self.crossover.do(self.problem, pop.X[None, parents, :], X=pop.X[[i], :])

            # evaluate the offspring
            F, _ = self.evaluator.eval(self.problem, X)

            # update the ideal point
            self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], F], axis=0), axis=0)

            # for each offspring that was created
            for k in range(self.crossover.n_children):
                # the weights of each neighbor
                weights_of_neighbors = self.weights[self.neighbours[i], :]

                # calculate the decomposed values for each neighbour
                FV = tchebi(pop.F[self.neighbours[i]], weights_of_neighbors, self.ideal_point)
                off_FV = tchebi(F[[k], :], weights_of_neighbors, self.ideal_point)

                # get the absolute index in F where offspring is better than the current F (decomposed space)
                off_is_better = self.neighbours[i][np.where(off_FV < FV)[0]]
                pop.F[off_is_better, :] = F[k, :]
                pop.X[off_is_better, :] = X[k, :]
