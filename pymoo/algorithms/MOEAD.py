import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.operators.crossover.real_differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymop.util import get_weights


class MOEAD(Algorithm):
    def __init__(self,
                 pop_size=100,
                 n_neighbors=20,
                 sampling=RealRandomSampling(),
                 mutation=PolynomialMutation(),
                 crossover=DifferentialEvolutionCrossover(),
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.sampling = sampling

        self.crossover = crossover
        self.mutation = mutation
        self.n_neighbors = n_neighbors

        # initialized when problem is known
        self.weights = None
        self.neighbours = None

    def _initialize(self, problem):
        self.weights = get_weights(self.pop_size, problem.n_obj, func_random=random.random, method="uniform")
        self.neighbours = np.argsort(cdist(self.weights, self.weights), axis=1)[:, :self.n_neighbors]

    def _solve(self, problem, evaluator):

        # create the population according to the factoring strategy
        if isinstance(self.sampling, np.ndarray):
            X = self.sampling
        else:
            X = self.sampling.sample(problem, self.pop_size, self)

        # evaluate and find ideal point
        F, _ = evaluator.eval(problem, X)
        ideal_point = np.min(F, axis=0)

        iteration = 1

        while evaluator.has_next():

            # iterate for each member of the population
            for i in range(self.pop_size):

                # select the parents from the neighbourhood
                parents = self.neighbours[i, random.perm(self.n_neighbors)[:self.crossover.n_parents]]

                # do recombination and create an offspring
                off_X = self.crossover.do(problem, X[None, parents,:], X=X[[i],:])

                # do the mutation
                off_X = self.mutation.do(problem, off_X)

                # evaluate the offspring
                off_F, _ = evaluator.eval(problem, off_X)

                # update the ideal point
                ideal_point = np.min(np.concatenate([ideal_point[None, :], off_F], axis=0), axis=0)

                # for each offspring that was created
                for k in range(self.crossover.n_children):

                    # the weights of each neighbor
                    weights_of_neighbors = self.weights[self.neighbours[i], :]

                    # calculate the decomposed values for each neighbour
                    FV = tchebi(F[self.neighbours[i]], weights_of_neighbors, ideal_point)
                    off_FV = tchebi(off_F[[k], :], weights_of_neighbors, ideal_point)

                    # get the absolute index in F where offspring is better than the current F (decomposed space)
                    off_is_better = self.neighbours[i][np.where(off_FV < FV)[0]]
                    F[off_is_better, :] = off_F[k, :]
                    X[off_is_better, :] = off_X[k, :]

            self._do_each_generation(iteration, evaluator, X, F)
            iteration += 1

        return X, F, np.zeros(self.pop_size)[:, None]

    def _do_each_generation(self, n_gen, evaluator, X, F):
        if self.verbose > 0:
            print('gen = %d' % (n_gen + 1))
        if self.verbose > 1:
            pass
        if self.callback is not None:
            pass
            #self.callback(self, evaluator.counter, pop)

        if self.history is not None:
            self.history.append(
                {'n_gen': n_gen,
                 'n_evals': evaluator.counter,
                 'X': np.copy(X),
                 'F': np.copy(F)
                 })


def tchebi(F, weights, ideal_point):
    v = np.abs((F - ideal_point) * weights)
    return np.max(v, axis=1)

