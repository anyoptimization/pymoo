import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.survival import Survival
from pymoo.operators.default_operators import set_default_if_none
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymop.util import get_uniform_weights


class MOEAD(GeneticAlgorithm):
    def __init__(self,
                 var_type="real",
                 n_neighbors=15,
                 **kwargs):

        #set_if_none(kwargs, "crossover", DifferentialEvolutionCrossover())
        set_default_if_none(var_type, kwargs)

        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

        # initialized when problem is known
        self.weights = None
        self.neighbours = None
        self.ideal_point = None

    def _initialize(self):

        # weights to be used for decomposition
        self.weights = get_uniform_weights(self.pop_size, self.problem.n_obj)

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbours = np.argsort(cdist(self.weights, self.weights), axis=1)[:, :self.n_neighbors + 1]

        # survival selection is included in the _mating method
        self.survival = Survival()

        pop = super()._initialize()

        # set the initial ideal point
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
            X = self.mutation.do(self.problem, X)

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

    def _display_attrs(self, D):
        return disp_multi_objective(self.problem, self.evaluator, D)


def tchebi(F, weights, ideal_point):
    v = np.abs((F - ideal_point) * weights)
    return np.max(v, axis=1)
