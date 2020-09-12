import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.aggregated_genetic_algorithm import AggregatedGeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none

import pandas as pd
from sklearn.cluster import KMeans

# =========================================================================================================
# Implementation
# =========================================================================================================

class OnlineClusterMOEAD(AggregatedGeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 cluster=KMeans,
                 number_of_clusters=2,
                 **kwargs):
        """

        MOEAD Algorithm.

        Parameters
        ----------
        ref_dirs
        n_neighbors
        decomposition
        prob_neighbor_mating
        display
        kwargs
        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition
        self.cluster = cluster
        self.number_of_clusters = number_of_clusters
        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

    def _initialize(self):

        if isinstance(self.decomposition, str):

            # set a string
            decomp = self.decomposition

            # for one or two objectives use tchebi otherwise pbi
            if decomp == 'auto':
                if self.problem.n_obj <= 2:
                    decomp = 'tchebi'
                else:
                    decomp = 'pbi'

            # set the decomposition object
            self._decomposition = get_decomposition(decomp)

        else:
            self._decomposition = self.decomposition

        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        X = np.array([individual.F for individual in self.pop])
        cluster = self.cluster(n_clusters=self.number_of_clusters)
        cluster.fit(X.T)
        transformation_matrix = self.get_transformation_matrix(cluster)
        self.ideal_point = np.dot(transformation_matrix, self.ideal_point)

        self.current_generation = 0

    def _next(self):
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop

        self.evaluate_population_in_original_objectives(pop)

        # generate cluster for current population
        X = np.array([individual.F for individual in self.pop])
        cluster = self.cluster(n_clusters=self.number_of_clusters)
        cluster.fit(X.T)
        transformation_matrix = self.get_transformation_matrix(cluster)
        print(transformation_matrix)
        print('Current generation:', self.current_generation)
        
        self.reduce_population(pop, transformation_matrix)
        # iterate for each member of the population in random order
        for i in np.random.permutation(len(pop)):

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            if np.random.random() < self.prob_neighbor_mating:
                parents = N[np.random.permutation(self.n_neighbors)][:crossover.n_parents]
            else:
                parents = np.random.permutation(self.pop_size)[:crossover.n_parents]

            # do recombination and create an offspring
            off = crossover.do(self.problem, pop, parents[None, :])
            off = mutation.do(self.problem, off)
            off = off[np.random.randint(0, len(off))]
            
            # repair first in case it is necessary - disabled if instance of NoRepair
            off = repair.do(self.problem, off, algorithm=self)

            # evaluate the offspring
            self.evaluator.eval(self.problem, off)

            # reduce objectives in offspring
            off.F = np.dot(transformation_matrix, off.F)
            
            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get("F"), weights=np.dot(transformation_matrix, self.ref_dirs[N, :].T).T, ideal_point=self.ideal_point, utopian_point=np.array([0,0]))
            off_FV = self._decomposition.do(off.F[None, :], np.dot(transformation_matrix, self.ref_dirs[N, :].T).T, ideal_point=self.ideal_point, utopian_point=np.array([0,0]))

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off
        self.current_generation += 1

    def get_transformation_matrix(self, cluster):
        return pd.get_dummies(cluster.labels_).T.values
    
    def reduce_population(self, population, transformation_matrix):
        for individual in population:
            individual.F = self.problem.evaluate(individual.get('X'))
            individual.F = np.dot(transformation_matrix, individual.F)
    
    def evaluate_population_in_original_objectives(self, population):
        for individual in population:
            individual.F = self.problem.evaluate(individual.get('X'))
# parse_doc_string(MOEAD.__init__)
